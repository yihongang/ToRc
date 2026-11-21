//! torc-inkling <model.torc> <tokens.int32|-> [n] [threads|auto] [dump=logits.f32] [ctx=N]: Inkling 2's forward pass,
//! one token at a time as torc-server runs it, writing every position's logits (float32, n x n_vocab, the format
//! torc-run writes and torc-kl compares). A tokens file of `-` runs `n` made-up ids instead (deterministic, in the
//! model's vocabulary): enough to time it or to compare two builds, with nothing to prepare. For a real prompt,
//! `torc-tok <model> <text file> <out.int32>` writes one. This is the CPU reference the NPU tier's routed experts are checked against,
//! and with TORC_ROUTES (or TORC_NPU_PROFILE_OUT) the run also writes the experts it routed to (npu.rs).

use model::pool;
use std::io::Write;
use std::time::Instant;

fn main() {
    model::set_npu_env();
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: torc-inkling <model.torc> <tokens.int32> [n] [threads|auto] [dump=logits.f32] [ctx=N]");
        std::process::exit(2);
    }
    let dump = args.iter().find_map(|a| a.strip_prefix("dump="));
    let ctx: usize = args.iter().find_map(|a| a.strip_prefix("ctx=")).and_then(|v| v.parse().ok()).unwrap_or(4096);
    // "-": no tokens file, just n made-up ids (the model's vocabulary, the same every run)
    let made_up = args[1] == "-";
    let ask = args.get(2).and_then(|s| s.parse::<usize>().ok());
    let mut tokens: Vec<u32> = if made_up {
        Vec::new()
    } else {
        let raw = std::fs::read(&args[1]).unwrap_or_else(|e| {
            eprintln!("cannot read the tokens file {}: {e}", args[1]);
            std::process::exit(2)
        });
        let mut v: Vec<u32> = raw.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        if let Some(n) = ask {
            v.truncate(n);
        }
        v
    };
    let asked: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);
    let threads = pool::init(asked).threads;
    eprintln!("threads: {threads}");
    let t0 = Instant::now();
    let mut m = model::Inkling::load(&args[0], ctx).unwrap_or_else(|e| {
        eprintln!("torc-inkling: {e}");
        std::process::exit(1)
    });
    eprintln!("loaded in {:.1} s", t0.elapsed().as_secs_f64());
    let nv = m.n_vocab();
    if made_up {
        let k = ask.unwrap_or(32).max(1);
        tokens = (0..k).map(|i| (997 + i * 7919) as u32 % nv as u32).collect();
        eprintln!("{k} made-up tokens (no tokens file asked for)");
    }
    let mut logits = vec![0.0f32; nv];
    let mut out: Vec<u8> = Vec::new();
    let t1 = Instant::now();
    for (p, &id) in tokens.iter().enumerate() {
        m.forward(id, p as u32, &mut logits).unwrap_or_else(|e| {
            eprintln!("torc-inkling: {e}");
            std::process::exit(1)
        });
        if dump.is_some() {
            for v in &logits {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        eprintln!("token {p}: {:.2} s", t1.elapsed().as_secs_f64());
    }
    let s = t1.elapsed().as_secs_f64();
    if let Some(p) = dump {
        // the tokens the dump holds, beside it: torc-kl reads a dump against a tokens file of the same length
        let mut toks = Vec::with_capacity(4 * tokens.len());
        for t in &tokens {
            toks.extend_from_slice(&t.to_le_bytes());
        }
        for (name, bytes) in [(p.to_string(), out.as_slice()), (format!("{p}.tokens"), toks.as_slice())] {
            match std::fs::File::create(&name) {
                Ok(mut f) => {
                    if let Err(e) = f.write_all(bytes) {
                        eprintln!("cannot write {name}: {e}");
                        std::process::exit(2);
                    }
                }
                Err(e) => {
                    eprintln!("cannot write {name}: {e}");
                    std::process::exit(2);
                }
            }
        }
        eprintln!("dump: {p} (and its tokens, {p}.tokens)");
    }
    // TORC_PROF=1: the phase tables, per generated token
    let ng = tokens.len().max(1) as f64;
    for (name, phases, table) in model::phase_tables() {
        let ns: Vec<u64> = table.iter().map(|a| a.load(std::sync::atomic::Ordering::Relaxed)).collect();
        if ns.iter().any(|&v| v > 0) {
            let total: u64 = ns.iter().sum();
            let parts: Vec<String> =
                phases.iter().zip(&ns).map(|(n, &v)| format!("{n} {:.2}", v as f64 / 1e6 / ng)).collect();
            eprintln!("{name}: {:.2} ms/token: {}", total as f64 / 1e6 / ng, parts.join(", "));
        }
    }
    if let Some((w, g)) = model::npu::stats() {
        eprintln!("npu: {g} graphs, {:.1} ms waited", w * 1e3);
    }
    model::npu::write_profile(m.h.n_layer, m.h.n_expert);
    let top = logits.iter().enumerate().fold((0, f32::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b });
    eprintln!("last: top {} ({:.4})", top.0, top.1);
    eprintln!("{} tokens: {s:.1} s = {:.2} tok/s", tokens.len(), tokens.len() as f64 / s);
}
