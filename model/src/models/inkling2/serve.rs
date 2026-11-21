//! serve.rs - Inkling 2 for torc-server (chat/serve.rs). The text model reads a prompt one token at a time and writes
//! a reply with the Qwen sampler; its chat format and tool calls are in chat.rs.

use super::chat;
use super::model::Inkling;
use crate::api::Chat;
use crate::models::qwen::sample::Sampling;
use crate::serve::{Reply, Serve, defused, defused_tools, sampling};
use crate::tok::Tokenizer;

struct Engine {
    m: Inkling,
    tok: Tokenizer,
    ctx: usize,
    nv: usize,
    name: String,
    stops: Vec<u32>,
    kept: usize,
    timed: (usize, f64, f64, usize, f64),
    replies: u64,
}

impl Engine {
    fn ids_of(&self, c: &Chat) -> Vec<u32> {
        let d: &dyn Fn(&str) -> std::borrow::Cow<str> = &|t| self.tok.defuse(t, chat::MARKERS);
        let msgs = defused(&c.msgs, d);
        let tools = defused_tools(&c.tools, d);
        let text = chat::prompt(&msgs, &tools, c.effort.as_deref(), c.think);
        self.tok.encode(&text, true).unwrap_or_default()
    }
}

impl Serve for Engine {
    fn name(&self) -> &str {
        &self.name
    }

    fn count(&self, c: &Chat) -> usize {
        self.ids_of(c).len()
    }

    fn reply(&mut self, c: &Chat, emit: &mut dyn FnMut(bool, &str) -> bool) -> format::Result<Reply> {
        let ids = self.ids_of(c);
        if ids.is_empty() {
            return Err(format::Error("the prompt is empty".into()));
        }
        if ids.len() >= self.ctx {
            return Err(crate::chat::serve::too_long(ids.len(), self.ctx));
        }
        let max = c.max.min(self.ctx - ids.len());
        let t0 = std::time::Instant::now();
        self.m.clear();
        let mut logits = vec![0.0f32; self.nv];
        for (p, &id) in ids.iter().enumerate() {
            self.m.forward(id, p as u32, &mut logits)?;
        }
        let read_secs = t0.elapsed().as_secs_f64();
        let mut sp: Sampling = sampling(c);
        let mut pos = ids.len() as u32;
        let mut n = 0usize;
        let mut bytes: Vec<u8> = Vec::new();
        let mut first = 0.0f64;
        let t1 = std::time::Instant::now();
        let mut finish = "length";
        let mut id = sp.pick(&logits);
        loop {
            if self.stops.contains(&id) {
                finish = "stop";
                break;
            }
            if n >= max {
                break;
            }
            if n == 0 {
                first = t1.elapsed().as_secs_f64();
            }
            bytes.extend(self.tok.decode(id));
            n += 1;
            if n >= max {
                finish = "length";
                break;
            }
            self.m.forward(id, pos, &mut logits)?;
            pos += 1;
            id = sp.pick(&logits);
        }
        let write_secs = t1.elapsed().as_secs_f64();
        let text = String::from_utf8_lossy(&bytes).into_owned();
        let (calls, kept, reasoning) = chat::read(&text);
        let reasoning_tokens = self.tok.encode(&reasoning, false).map_or(0, |v| v.len());
        if !reasoning.is_empty() {
            let _ = emit(true, &reasoning);
        }
        if !kept.is_empty() {
            let _ = emit(false, &kept);
        }
        if !calls.is_empty() && finish == "stop" {
            finish = "tool_calls";
        }
        self.kept = 0;
        self.replies += 1;
        // the experts used so far go into this machine's usage file, so the next load can put the hottest on the NPU
        // (npu.rs: with no expert cache the tier counts the routes itself)
        match &self.m.w.cache {
            Some(c) => c.usage_flush(),
            None => crate::npu::usage_flush(&self.m.w, self.m.h.n_layer, self.m.h.n_expert),
        }
        let reply = Reply { prompt: ids.len(), output: n, reasoning: reasoning_tokens, finish, calls };
        self.timed = (ids.len(), read_secs, first, n, write_secs);
        Ok(reply)
    }

    fn kept(&self) -> usize {
        self.kept
    }

    fn timed(&self) -> (usize, f64, f64, usize, f64) {
        self.timed
    }

    fn replies(&self) -> u64 {
        self.replies
    }
}

/// Loads an Inkling 2 file for torc-server.
pub fn open(path: &str, ctx: usize, name: Option<String>) -> Box<dyn Serve> {
    // on one machine the NPU's hot experts come from this machine's own use of the model (npu.rs)
    crate::npu::own_usage(!crate::switches::given("TORC_PIPE"));
    let m = Inkling::load(path, ctx).unwrap_or_else(|e| {
        eprintln!("torc-server: {e}");
        std::process::exit(1)
    });
    let tok = Tokenizer::open(path).unwrap_or_else(|e| {
        eprintln!("torc-server: {e}");
        std::process::exit(1)
    });
    let nv = m.n_vocab();
    let meta = format::read::Model::open(path).ok().and_then(|f| format::meta::Meta::parse(&f.metadata).ok());
    let mut stops: Vec<u32> = Vec::new();
    if let Some(e) = meta.as_ref().and_then(|mm| mm.u32("tokenizer.ggml.eos_token_id")) {
        stops.push(e);
    }
    if let Some(e) = tok.find(chat::END) {
        stops.push(e);
    }
    stops.sort_unstable();
    stops.dedup();
    let name = name.unwrap_or_else(|| m.name.clone());
    Box::new(Engine { m, tok, ctx, nv, name, stops, kept: 0, timed: (0, 0.0, 0.0, 0, 0.0), replies: 0 })
}
