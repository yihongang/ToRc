//! model.rs - Inkling 2's text forward pass.
//!
//! The model is a 24-layer decoder whose middle 12 layers are visited three times, with a relative-position bias instead
//! of RoPE, short causal convolutions on K, V, the attention output and the FFN output, and attention-residual blocks
//! every six layers. This port follows llama.cpp's `inkling2` / `inkling2_loop` graph and the MLX reference that came
//! with it. It runs one token at a time; the KV cache and the short-convolution windows are kept per layer visit.

use super::hparams::Hp;
use crate::attn::KvCache;
use crate::moe::Mat;
use crate::ops::sigmoid;
use crate::weights::{TensorRef, Weights};
use format::{Result, err};
use kernels::{Q8, QK, ops as kops};

/// TORC_PROF=1 phase times for Inkling 2's forward pass, summed over every layer visit and every generated token:
/// where a token's time goes. `torc-inkling` prints the table; `-` phases are the ones a run does not use.
pub static INK_PHASES: [&str; 11] =
    ["embed", "inputs", "qkv", "conv k/v", "attend", "attn out", "ffn mix", "experts", "ffn conv", "final mix", "head"];
pub static INK_NS: [std::sync::atomic::AtomicU64; 11] = [const { std::sync::atomic::AtomicU64::new(0) }; 11];

/// A matrix-vector product's input quantised to Q8_K, reused across calls.
struct Q8Buf(Q8);

impl Q8Buf {
    fn new(d_model: usize) -> Q8Buf {
        Q8Buf(Q8::new(d_model.div_ceil(QK)))
    }
    fn quant(&mut self, x: &[f32]) -> kernels::Q8s<'_> {
        assert!(x.len().is_multiple_of(QK));
        let n = x.len() / QK;
        assert!(n <= self.0.len(), "quantised buffer too small");
        let q = self.0.sub_mut(0..n);
        kops::q8k_quantize(x, q);
        self.0.sub(0..n)
    }
}

fn rms(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    crate::ops::rms_norm(&mut out, x, w, eps);
    out
}

fn unit_rms(x: &[f32], eps: f32) -> Vec<f32> {
    let mut ss = 0.0f64;
    for &v in x {
        ss += v as f64 * v as f64;
    }
    let s = 1.0f32 / ((ss / x.len() as f64) as f32 + eps).sqrt();
    x.iter().map(|&v| v * s).collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    kops::dot_f32(a, b)
}

/// `cur` with each row scaled down to an L2 norm of at most `threshold`, after an optional per-channel scale.
fn clip(cur: &mut [f32], scale: Option<&[f32]>, threshold: f32) {
    if let Some(s) = scale {
        for (v, &s) in cur.iter_mut().zip(s) {
            *v *= s;
        }
    }
    let norm = cur.iter().map(|&v| v * v).sum::<f32>().sqrt() + 1e-6;
    let f = (threshold / norm).min(1.0);
    if f < 1.0 {
        for v in cur.iter_mut() {
            *v *= f;
        }
    }
}

/// One visit's short-convolution history: the last `k - 1` inputs of each of the four convolutions.
#[derive(Default)]
struct ConvState {
    rows: [Vec<Vec<f32>>; 4],
}

impl ConvState {
    /// Causal depthwise convolution with a residual, as llama.cpp's `convolve`: `which` 0 K, 1 V, 2 attention out, 3
    /// FFN out. `dynamic` is the input-dependent taps for a full-attention K or V convolution (head-major, `k` a head).
    fn run(
        &mut self,
        which: usize,
        cur: &[f32],
        weight: &[f32],
        dynamic: Option<&[f32]>,
        heads: usize,
        k: usize,
    ) -> Vec<f32> {
        let hist = &mut self.rows[which];
        // the state starts as k - 1 rows of zeros, so the first token's window is causal padding
        let zeros = vec![vec![0.0f32; cur.len()]; (k - 1).saturating_sub(hist.len())];
        let mut xp: Vec<&[f32]> = zeros.iter().map(Vec::as_slice).chain(hist.iter().map(Vec::as_slice)).collect();
        xp.push(cur);
        let mut out = vec![0.0f32; cur.len()];
        let dynamic = dynamic.filter(|d| !d.is_empty());
        for c in 0..cur.len() {
            let mut v = 0.0f32;
            if let Some(dynw) = dynamic {
                let head = c / (cur.len() / heads.max(1));
                for lag in 0..k {
                    let tap = weight[c * k + lag] + dynw[head * k + lag];
                    v += tap * xp[k - 1 - lag][c];
                }
            } else {
                for lag in 0..k {
                    v += weight[c * k + lag] * xp[lag][c];
                }
            }
            out[c] = (cur[c] + v) * (1.0 / (((if dynamic.is_some() { 2 } else { 1 }) * k + 1) as f32).sqrt());
        }
        hist.push(cur.to_vec());
        if hist.len() > k - 1 {
            hist.remove(0);
        }
        out
    }
}

/// Softmax over depth sources: `blocks` followed by the current partial sum, mixed by `query` (`q` rows).
fn mix(blocks: &[Vec<f32>], cur: &[f32], query: &[f32], q: usize, eps: f32) -> Vec<f32> {
    let dm = cur.len();
    let n = blocks.len() + 1;
    let mut src: Vec<Vec<f32>> = blocks.iter().map(|b| unit_rms(b, eps)).collect();
    src.push(unit_rms(cur, eps));
    let mut out = vec![0.0f32; q * dm];
    let scale = 1.0 / (dm as f32).sqrt();
    for qi in 0..q {
        let qv = &query[qi * dm..(qi + 1) * dm];
        let mut scores = vec![0.0f32; n];
        let mut m = f32::NEG_INFINITY;
        for (s, b) in src.iter().enumerate() {
            let d = dot(qv, b) * scale;
            scores[s] = d;
            if d > m {
                m = d;
            }
        }
        let mut z = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - m).exp();
            z += *s;
        }
        let o = &mut out[qi * dm..(qi + 1) * dm];
        for (s, b) in src.iter().enumerate() {
            let w = scores[s] / z;
            for i in 0..dm {
                o[i] += w * b[i];
            }
        }
    }
    out
}

/// The loop controller: `(1 - tanh(forget)) * x + tanh(gain) * message`.
fn control(
    cur: &[f32],
    message: &[f32],
    down: &[f32],
    gain: &[f32],
    forget: &[f32],
    rank: usize,
    eps: f32,
) -> Vec<f32> {
    let dm = cur.len();
    let mut g = Vec::with_capacity(2 * dm);
    g.extend(unit_rms(cur, eps));
    g.extend(unit_rms(message, eps));
    let mut low = vec![0.0f32; rank];
    for (r, o) in low.iter_mut().enumerate() {
        *o = dot(&down[r * 2 * dm..(r + 1) * 2 * dm], &g);
    }
    let mut out = vec![0.0f32; dm];
    for i in 0..dm {
        let ga = dot(&gain[i * rank..(i + 1) * rank], &low).tanh();
        let fo = dot(&forget[i * rank..(i + 1) * rank], &low).tanh();
        out[i] = (1.0 - fo) * cur[i] + ga * message[i];
    }
    out
}

/// SiTU: `beta*tanh(gate/beta)*sigmoid(gate) * lb*tanh(up/lb)`.
fn situ(gate: f32, up: f32, beta: f32, lb: f32) -> f32 {
    (beta * (gate / beta).tanh() * sigmoid(gate)) * (lb * (up / lb).tanh())
}

/// A quantised matrix-vector product. The input is quantised to Q8_K, which is what ToRc's dense kernels take.
fn matvec(t: &TensorRef<'_>, x: &[f32], qbuf: &mut Q8Buf, out: &mut [f32]) {
    let m = Mat::of(t);
    let q = qbuf.quant(x);
    m.apply(q, out);
}

/// A small resident tensor as f32: the float types as they are, and a quantised one (the mainline file stores
/// `attn_sink_proj` as Q8_0) through the dequantiser.
fn to_f32(t: &TensorRef<'_>) -> Result<Vec<f32>> {
    match t.t.ty {
        0 | 1 | 30 => t.f32s(),
        ty => {
            let (cols, rows) = t.shape();
            let mut y = vec![0.0f32; cols * rows];
            kops::dequantize(ty, t.data, &mut y);
            Ok(y)
        }
    }
}

/// The layers' resident small tensors, widened to f32 at load.
struct LayerW {
    attn_norm: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    v_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    /// per head, head_dim values
    q_clip: Vec<f32>,
    /// per head, rel_len values
    rel_clip: Vec<f32>,
    /// per distance, rel_len values
    rel_bias: Vec<f32>,
    /// per head, a d_model row
    sink_proj: Vec<f32>,
    /// channel-major, k taps each
    k_conv: Vec<f32>,
    v_conv: Vec<f32>,
    a_conv: Vec<f32>,
    f_conv: Vec<f32>,
    /// q rows of d_model
    attn_res_query: Vec<f32>,
    ffn_res_query: Vec<f32>,
    router: Vec<f32>,
    exp_probs_b: Vec<f32>,
    loop_ctrl: Option<LoopW>,
}

struct LoopW {
    msg_query: Vec<f32>,
    /// four controllers; each `rank x 2*d_model`, `d_model x rank`, `d_model x rank`
    down: Vec<f32>,
    gain: Vec<f32>,
    forget: Vec<f32>,
}

pub struct Inkling {
    pub w: Weights,
    pub h: Hp,
    pub name: String,
    layers: Vec<std::sync::Arc<LayerW>>,
    groups: Vec<u32>,
    tok_norm: Vec<f32>,
    out_norm: Vec<f32>,
    out_query: Vec<f32>,
    kv: Vec<KvCache>,
    conv: Vec<ConvState>,
    ctx: usize,
}

impl Inkling {
    pub fn load(path: &str, n_ctx: usize) -> Result<Inkling> {
        let w = Weights::load(path)?;
        let h = Hp::from_file(&w.file)?;
        if !w.file.experts.is_empty() && h.n_used > w.file.experts[0].n_expert {
            return err("inkling2_loop: expert_used_count is larger than the experts");
        }
        let mut groups = vec![u32::MAX; h.n_layer as usize];
        for (i, x) in w.file.experts.iter().enumerate() {
            if let Some(g) = groups.get_mut(x.layer as usize) {
                *g = i as u32;
            }
        }
        if groups.iter().any(|&g| g == u32::MAX) {
            return err("inkling2_loop: a layer has no routed experts");
        }
        let mut layers = Vec::with_capacity(h.n_layer as usize);
        for l in 0..h.n_layer {
            layers.push(std::sync::Arc::new(LayerW::load(&w, &h, l)?));
        }
        // TORC_NPU=1: the routed experts' hot set goes on the NPU (npu.rs). The attention products, the short
        // convolutions and the loop control stay on the CPU, so only the experts are uploaded
        crate::npu::load_experts_only(&w, h.n_layer, &groups);
        if crate::npu::up() && crate::npu::min_tokens() > 1 {
            // one token is all this model ever has; at that size the NPU measured 0.27x the CPU (AGENTS.md section 5)
            eprintln!(
                "npu: Inkling 2 reads one token at a time; its experts measured slower on the NPU (TORC_NPU_MIN_N=1 would take them)"
            );
        }
        let tok_norm = to_f32(&w.need("token_embd_norm.weight")?)?;
        let out_norm = to_f32(&w.need("output_norm.weight")?)?;
        let out_query = to_f32(&w.need("output_res_query.weight")?)?;
        if tok_norm.len() != h.n_embd as usize
            || out_norm.len() != h.n_embd as usize
            || out_query.len() != h.n_embd as usize
        {
            return err("inkling2_loop: a final norm or query has the wrong width");
        }
        let ctx = n_ctx.max(1);
        let mut kv = Vec::with_capacity(h.n_slot as usize);
        for _ in 0..h.n_slot {
            kv.push(KvCache::with(h.n_head_kv as usize, h.head_dim as usize, ctx, false));
        }
        let conv = (0..h.n_slot).map(|_| ConvState::default()).collect();
        let name = format::meta::Meta::parse(&w.file.metadata)?.str("general.name").unwrap_or("Inkling 2").to_string();
        Ok(Inkling { w, h, name, layers, groups, tok_norm, out_norm, out_query, kv, conv, ctx })
    }

    pub fn n_vocab(&self) -> usize {
        self.h.n_vocab_unpadded as usize
    }

    pub fn ctx(&self) -> usize {
        self.ctx
    }

    /// Forgets every cached position and convolution window.
    pub fn clear(&mut self) {
        for kv in &mut self.kv {
            kv.k_scale.fill(0.0);
            kv.v_scale.fill(0.0);
            kv.k.fill(0);
            kv.v.fill(0);
        }
        for c in &mut self.conv {
            for r in &mut c.rows {
                r.clear();
            }
        }
    }

    fn embed(&self, token: u32) -> Result<Vec<f32>> {
        let t = self.w.need("token_embd.weight")?;
        let (cols, rows) = t.shape();
        if token as usize >= rows {
            return err(format!("token {token} is past the embedding's {rows} rows"));
        }
        let rb = t.row_bytes();
        let mut x = vec![0.0f32; cols];
        let row = &t.data[token as usize * rb..(token as usize + 1) * rb];
        match t.t.ty {
            0 => {
                for (o, c) in x.iter_mut().zip(row.chunks_exact(4)) {
                    *o = f32::from_le_bytes(c.try_into().unwrap());
                }
            }
            1 => {
                for (o, c) in x.iter_mut().zip(row.chunks_exact(2)) {
                    *o = kernels::f16_to_f32(u16::from_le_bytes(c.try_into().unwrap()));
                }
            }
            30 => {
                for (o, c) in x.iter_mut().zip(row.chunks_exact(2)) {
                    *o = f32::from_bits((u16::from_le_bytes(c.try_into().unwrap()) as u32) << 16);
                }
            }
            ty => kops::dequantize(ty, row, &mut x),
        }
        Ok(rms(&x, &self.tok_norm, self.h.eps))
    }

    /// The forward pass for one token at `pos`; `logits` is `n_vocab_unpadded` long.
    pub fn forward(&mut self, token: u32, pos: u32, logits: &mut [f32]) -> Result<()> {
        if pos as usize >= self.ctx {
            return err(format!("position {pos} is past the context of {}", self.ctx));
        }
        let mut l = crate::Laps::start(&INK_NS);
        let mut cur = self.embed(token)?;
        l.lap(0);
        let mut blocks: Vec<Vec<f32>> = Vec::new();
        let mut prelude: Option<(Vec<f32>, Vec<Vec<f32>>)> = None;
        let mut bank: Option<(Vec<Vec<f32>>, Vec<f32>)> = None;
        let width = self.w.file.experts[0].width as usize;
        let max_q = (self.h.n_embd as usize).max(self.h.n_head as usize * self.h.head_dim as usize).max(width);
        let mut qbuf = Q8Buf::new(max_q);
        for (visit, ip, pass) in self.h.schedule() {
            if ip == self.h.core_start {
                if prelude.is_none() {
                    prelude = Some((cur.clone(), blocks.clone()));
                } else {
                    let pb = prelude.as_ref().unwrap().1.len();
                    bank = Some((blocks[pb + 1..].to_vec(), cur.clone()));
                    let (p, pb) = prelude.as_ref().unwrap();
                    cur = p.clone();
                    blocks = pb.clone();
                }
            }
            let messages = if pass > 0 {
                let (b, o) = bank.as_ref().expect("bank for a loop pass");
                let lw = self.layers[self.h.layer_of(visit) as usize].clone();
                Some(mix(b, o, &lw.loop_ctrl.as_ref().unwrap().msg_query, 4, self.h.eps))
            } else {
                None
            };
            cur = self.layer(visit, ip, cur, &mut blocks, messages.as_ref(), pos, &mut qbuf, &mut l)?;
        }
        let mixed = mix(&blocks, &cur, &self.out_query, 1, self.h.eps);
        let x = rms(&mixed[..self.h.n_embd as usize], &self.out_norm, self.h.eps);
        l.lap(9);
        self.head(&x, logits, &mut qbuf)?;
        l.lap(10);
        if self.h.softcap > 0.0 {
            let c = self.h.softcap;
            for l in logits.iter_mut() {
                *l = c * (*l / c).tanh();
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn layer(
        &mut self,
        visit: u32,
        ip: u32,
        mut cur: Vec<f32>,
        blocks: &mut Vec<Vec<f32>>,
        messages: Option<&Vec<f32>>,
        pos: u32,
        qbuf: &mut Q8Buf,
        lap: &mut crate::Laps,
    ) -> Result<Vec<f32>> {
        let l = self.h.layer_of(visit);
        let lw = self.layers[l as usize].clone();
        let is_swa = self.h.swa[ip as usize];
        let dm = self.h.n_embd as usize;
        let hd = self.h.head_dim as usize;
        let nh = self.h.n_head as usize;
        let nkv = self.h.n_head_kv as usize;
        let dkv = nkv * hd;
        let k = self.h.n_conv as usize;
        let inp_sa = cur.clone();
        let mut inputs = [cur.clone(), cur.clone(), cur.clone()];
        if ip > 0 {
            let q = if is_swa { 1 } else { 3 };
            let mixed = mix(blocks, &cur, &lw.attn_res_query, q, self.h.eps);
            if is_swa {
                inputs = [mixed[..dm].to_vec(), mixed[..dm].to_vec(), mixed[..dm].to_vec()];
            } else {
                inputs = [mixed[..dm].to_vec(), mixed[dm..2 * dm].to_vec(), mixed[2 * dm..3 * dm].to_vec()];
            }
        }
        if let Some(m) = messages {
            for (j, x) in inputs.iter_mut().enumerate() {
                let c = &lw.loop_ctrl.as_ref().unwrap();
                let r = self.h.rank as usize;
                let d = &c.down[j * r * 2 * dm..(j + 1) * r * 2 * dm];
                let g = &c.gain[j * dm * r..(j + 1) * dm * r];
                let f = &c.forget[j * dm * r..(j + 1) * dm * r];
                *x = control(x, &m[j * dm..(j + 1) * dm], d, g, f, r, self.h.eps);
            }
        }
        if is_swa && messages.is_some() {
            for x in inputs.iter_mut() {
                *x = rms(x, &lw.attn_norm, self.h.eps);
            }
        } else if is_swa {
            let n = rms(&inputs[0], &lw.attn_norm, self.h.eps);
            inputs = [n.clone(), n.clone(), n];
        } else {
            inputs[0] = rms(&inputs[0], &lw.q_norm, self.h.eps);
            inputs[1] = rms(&inputs[1], &lw.k_norm, self.h.eps);
            inputs[2] = rms(&inputs[2], &lw.v_norm, self.h.eps);
        }

        lap.lap(1);
        let mut q = vec![0.0f32; nh * hd];
        let mut kcur = vec![0.0f32; dkv];
        let mut vcur = vec![0.0f32; dkv];
        matvec(&self.w.layer(l, "attn_q.weight")?, &inputs[0], qbuf, &mut q);
        matvec(&self.w.layer(l, "attn_k.weight")?, &inputs[1], qbuf, &mut kcur);
        matvec(&self.w.layer(l, "attn_v.weight")?, &inputs[2], qbuf, &mut vcur);
        let mut r = vec![0.0f32; nh * self.h.rel_len as usize];
        matvec(&self.w.layer(l, "attn_rel.weight")?, &inputs[0], qbuf, &mut r);
        let dyn_k = if is_swa {
            None
        } else {
            let mut d = vec![0.0f32; k * nkv];
            matvec(&self.w.layer(l, "attn_k_conv_proj.weight")?, &inputs[1], qbuf, &mut d);
            Some(d)
        };
        let dyn_v = if is_swa {
            None
        } else {
            let mut d = vec![0.0f32; k * nkv];
            matvec(&self.w.layer(l, "attn_v_conv_proj.weight")?, &inputs[2], qbuf, &mut d);
            Some(d)
        };
        lap.lap(2);
        let mut kk = {
            let conv = &mut self.conv[visit as usize];
            conv.run(0, &kcur, &lw.k_conv, dyn_k.as_deref(), nkv, k)
        };
        let mut vv = {
            let conv = &mut self.conv[visit as usize];
            conv.run(1, &vcur, &lw.v_conv, dyn_v.as_deref(), nkv, k)
        };
        for h in 0..nh {
            clip(&mut q[h * hd..(h + 1) * hd], Some(&lw.q_clip[h * hd..(h + 1) * hd]), self.h.qk_clip);
        }
        for h in 0..nkv {
            clip(&mut kk[h * hd..(h + 1) * hd], None, self.h.qk_clip);
        }
        for h in 0..nkv {
            let n = unit_rms(&vv[h * hd..(h + 1) * hd], self.h.eps);
            vv[h * hd..(h + 1) * hd].copy_from_slice(&n);
        }
        for h in 0..nh {
            clip(
                &mut r[h * self.h.rel_len as usize..(h + 1) * self.h.rel_len as usize],
                Some(&lw.rel_clip[h * self.h.rel_len as usize..(h + 1) * self.h.rel_len as usize]),
                self.h.rel_clip,
            );
        }
        if !is_swa {
            let tau = if pos as u64 + 1 > self.h.log_floor as u64 {
                1.0 + self.h.log_alpha * (((pos as f32 + 1.0) / self.h.log_floor as f32).ln())
            } else {
                1.0
            };
            for v in q.iter_mut() {
                *v *= tau;
            }
            for v in r.iter_mut() {
                *v *= tau;
            }
        }
        lap.lap(3);
        let mut gate = vec![0.0f32; nh * hd];
        matvec(&self.w.layer(l, "attn_gate.weight")?, &inputs[0], qbuf, &mut gate);
        let mut sink = vec![0.0f32; nh];
        for (hh, s) in sink.iter_mut().enumerate() {
            *s = dot(&lw.sink_proj[hh * dm..(hh + 1) * dm], &inputs[0]);
        }
        let att = self.attend(visit, &q, &kk, &vv, &r, &sink, &lw.rel_bias, pos, is_swa)?;
        let mut out = vec![0.0f32; nh * hd];
        for i in 0..nh * hd {
            out[i] = att[i] * sigmoid(gate[i]);
        }
        lap.lap(4);
        let mut o = vec![0.0f32; dm];
        matvec(&self.w.layer(l, "attn_output.weight")?, &out, qbuf, &mut o);
        let a = {
            let conv = &mut self.conv[visit as usize];
            conv.run(2, &o, &lw.a_conv, None, nkv, k)
        };
        if ip.is_multiple_of(self.h.block) {
            blocks.push(inp_sa);
            cur = a;
        } else {
            cur = inp_sa.iter().zip(&a).map(|(x, y)| x + y).collect();
        }
        lap.lap(5);
        let ffn_inp = cur.clone();
        let mut ffn = mix(blocks, &cur, &lw.ffn_res_query, 1, self.h.eps)[..dm].to_vec();
        if let Some(m) = messages {
            let c = lw.loop_ctrl.as_ref().unwrap();
            let rk = self.h.rank as usize;
            ffn = control(
                &ffn,
                &m[3 * dm..4 * dm],
                &c.down[3 * rk * 2 * dm..4 * rk * 2 * dm],
                &c.gain[3 * dm * rk..4 * dm * rk],
                &c.forget[3 * dm * rk..4 * dm * rk],
                rk,
                self.h.eps,
            );
        }
        ffn = rms(&ffn, &lw.ffn_norm, self.h.eps);
        lap.lap(6);
        let m = self.moe(l, &ffn, qbuf)?;
        lap.lap(7);
        let m = {
            let conv = &mut self.conv[visit as usize];
            conv.run(3, &m, &lw.f_conv, None, nkv, k)
        };
        let r = ffn_inp.iter().zip(&m).map(|(x, y)| x + y).collect();
        lap.lap(8);
        Ok(r)
    }

    #[allow(clippy::too_many_arguments)]
    fn attend(
        &mut self,
        visit: u32,
        q: &[f32],
        kk: &[f32],
        vv: &[f32],
        r: &[f32],
        sink: &[f32],
        rel_bias: &[f32],
        pos: u32,
        is_swa: bool,
    ) -> Result<Vec<f32>> {
        let hd = self.h.head_dim as usize;
        let nh = self.h.n_head as usize;
        let nkv = self.h.n_head_kv as usize;
        let rel_len = self.h.rel_len as usize;
        let extent = self.h.rel_extent as usize;
        let cap = self.ctx;
        {
            let kv = &mut self.kv[visit as usize];
            for h in 0..nkv {
                let at = (h * cap + pos as usize) * hd;
                let mut tmp = vec![0i8; hd];
                kv.k_scale[h * cap + pos as usize] = crate::attn::quant8(&kk[h * hd..(h + 1) * hd], &mut tmp);
                kv.k[at..at + hd].copy_from_slice(&tmp);
                kv.v_scale[h * cap + pos as usize] = crate::attn::quant8(&vv[h * hd..(h + 1) * hd], &mut tmp);
                kv.v[at..at + hd].copy_from_slice(&tmp);
            }
        }
        // the relative bias table: per head, a value for each distance in 0..extent
        let mut table = vec![0.0f32; nh * extent];
        for h in 0..nh {
            let rh = &r[h * rel_len..(h + 1) * rel_len];
            for d in 0..extent {
                table[h * extent + d] = dot(rh, &rel_bias[d * rel_len..(d + 1) * rel_len]);
            }
        }
        let lo = if is_swa { (pos as usize + 1).saturating_sub(self.h.window as usize) } else { 0 };
        let mut out = vec![0.0f32; nh * hd];
        let mut kf = vec![0.0f32; hd];
        let mut vf = vec![0.0f32; hd];
        for h in 0..nh {
            let head = h * hd..(h + 1) * hd;
            let kvh = (h * nkv / nh).min(nkv - 1);
            let mut maxs = 0.0f32;
            let mut sum = 1.0f32;
            let mut acc = vec![0.0f32; hd];
            for j in lo..=pos as usize {
                let at = (kvh * cap + j) * hd;
                let ks = self.kv[visit as usize].k_scale[kvh * cap + j];
                let vs = self.kv[visit as usize].v_scale[kvh * cap + j];
                for i in 0..hd {
                    kf[i] = self.kv[visit as usize].k[at + i] as f32 * ks;
                    vf[i] = self.kv[visit as usize].v[at + i] as f32 * vs;
                }
                let dist = pos as usize - j;
                let bias = if dist < extent { table[h * extent + dist] } else { 0.0 };
                let score = dot(&q[head.clone()], &kf) * (1.0 / (hd as f32).sqrt()) + bias - sink[h];
                if score > maxs {
                    let f = (maxs - score).exp();
                    sum *= f;
                    for a in acc.iter_mut() {
                        *a *= f;
                    }
                    maxs = score;
                }
                let w = (score - maxs).exp();
                sum += w;
                for (a, v) in acc.iter_mut().zip(&vf) {
                    *a += w * v;
                }
            }
            for (o, a) in out[head].iter_mut().zip(&acc) {
                *o = a / sum;
            }
        }
        Ok(out)
    }

    /// The router's sigmoid probabilities, the top-k over them with their bias, and the routing sum.
    fn route(&self, l: u32, x: &[f32]) -> (Vec<f32>, f32, Vec<(u32, f32)>) {
        let dm = self.h.n_embd as usize;
        let lw = &self.layers[l as usize];
        let mut logits = vec![0.0f32; self.h.n_expert as usize];
        for (e, o) in logits.iter_mut().enumerate() {
            *o = dot(&lw.router[e * dm..(e + 1) * dm], x);
        }
        let probs = logits.iter().map(|&v| sigmoid(v)).collect::<Vec<_>>();
        let sel: Vec<f32> = probs.iter().zip(&lw.exp_probs_b).map(|(p, b)| p + b).collect();
        let mut chosen = crate::ops::top_k(&sel, self.h.n_used as usize);
        chosen.sort_unstable_by_key(|x| x.0);
        let sum: f32 = chosen.iter().map(|&(e, _)| probs[e as usize]).sum::<f32>() + 1e-20;
        (probs, sum, chosen)
    }

    fn moe(&self, l: u32, x: &[f32], qbuf: &mut Q8Buf) -> Result<Vec<f32>> {
        let (probs, sum, chosen) = self.route(l, x);
        // this token's routes for the NPU tier's expert-use count (npu.rs; the usage file and TORC_ROUTES)
        if crate::npu::counting_on() {
            let r: Vec<(usize, f32)> = chosen.iter().map(|&(e, p)| (e as usize, p)).collect();
            crate::npu::record(l, std::slice::from_ref(&r));
        }
        // the NPU tier: an expert of the layer's hot set has its gate and up run on the NPU, its SiTU on the CPU and
        // its down on the NPU again, while the CPU runs the experts the NPU does not hold. With no hot expert chosen
        // (or the tier down) this is the CPU path exactly as it was.
        let hot: Vec<Option<[i32; 3]>> = chosen.iter().map(|&(e, _)| crate::npu::hot_of(l, e as usize, 1)).collect();
        if hot.iter().any(Option::is_some) {
            return self.moe_staged(l, x, qbuf, &chosen, &probs, sum, &hot);
        }
        self.moe_cpu(l, x, qbuf, &chosen, &probs, sum, &[])
    }

    /// The CPU's share of a token's route: each chosen expert's gate, up, SiTU and down, added to the output with its
    /// routing weight, in increasing expert id. This is the path the model has always taken; `skip` marks the experts
    /// the NPU took (the staged path's, which run there instead).
    ///
    /// One expert's matrices are 1.6 MiB, under `Mat::apply`'s 4 MiB rule, so an expert at a time would keep the whole
    /// MoE on one core (about 1.8 ms a visit on a Snapdragon X, against 0.7 ms over the pool). The experts of a token
    /// are independent, so they go over the pool one thread each, with the same `Mat::apply` calls an expert at a time
    /// would make: every row is the same dot product whichever thread computes it, and the sum is still made in
    /// increasing expert id below, so the bits are the CPU path's.
    fn moe_cpu(
        &self,
        l: u32,
        x: &[f32],
        qbuf: &mut Q8Buf,
        chosen: &[(u32, f32)],
        probs: &[f32],
        sum: f32,
        skip: &[bool],
    ) -> Result<Vec<f32>> {
        let dm = self.h.n_embd as usize;
        let group = self.groups[l as usize];
        let width = self.w.layer_experts(group).width as usize;
        let (beta, lb) = (self.h.situ_beta, self.h.situ_linear_beta);
        let mut out = vec![0.0f32; dm];
        // the experts this path runs, in increasing id
        let idx: Vec<usize> = (0..chosen.len()).filter(|&i| !skip.get(i).copied().unwrap_or(false)).collect();
        if idx.is_empty() {
            return Ok(out);
        }
        // the expert cache keeps only its layer's slots live, so its experts are asked for a group at a time and
        // computed before the next ask (without one, all of them come at once from the mapping)
        let per = self.w.cache.as_ref().map_or(usize::MAX, |c| c.layer_slots(l).clamp(1, 64));
        let kk = idx.len();
        // TORC_INKLING_MOE_PAR=0: one expert at a time on the calling thread, the way the model ran before the
        // experts were spread over the pool (an A/B). One expert has nothing to spread
        static PAR: std::sync::LazyLock<bool> =
            std::sync::LazyLock::new(|| crate::switches::get("TORC_INKLING_MOE_PAR").is_none_or(|v| v != "0"));
        if kk == 1 || !*PAR {
            let mut g = vec![0.0f32; width];
            let mut u = vec![0.0f32; width];
            let mut h = vec![0.0f32; width];
            let mut d = vec![0.0f32; dm];
            for &i in &idx {
                let e = chosen[i].0 as usize;
                let weight = probs[e] / sum * self.h.route_scale;
                self.expert_cpu(l, e, weight, x, qbuf, &mut out, &mut g, &mut u, &mut h, &mut d)?;
            }
            return Ok(out);
        }
        let mut gu = crate::scratch::buf::<f32>("inkling.moe_gu", 2 * kk * width);
        let mut hh = crate::scratch::buf::<f32>("inkling.moe_h", kk * width);
        let mut dd = crate::scratch::buf::<f32>("inkling.moe_d", kk * dm);
        for (c0, chunk) in idx.chunks(per).enumerate() {
            let ids: Vec<u32> = chunk.iter().map(|&i| chosen[i].0).collect();
            let packs = self.w.experts_of(l, group, &ids)?;
            let mats: Vec<[Mat; 3]> = packs
                .iter()
                .map(|&(data, p)| {
                    [&p.gate, &p.up, &p.down].map(|q| Mat {
                        ty: q.ty,
                        rows: q.rows as usize,
                        cols: q.cols as usize,
                        data: q.of(data),
                    })
                })
                .collect();
            let (g0, h0, d0) = (2 * c0 * width, c0 * width, c0 * dm);
            let (gp, hp, dp) = (
                crate::pool::SyncPtr::new(&mut gu),
                crate::pool::SyncPtr::new(&mut hh),
                crate::pool::SyncPtr::new(&mut dd),
            );
            let (gp, hp, dp) = (&gp, &hp, &dp);
            crate::pool::par_for(mats.len(), &|j| {
                // each job quantises its own copies of the token's vector and of its hidden, from this thread's own
                // scratch, so nothing is shared but the read-only x
                let mut qb = crate::scratch::q8("inkling.moe_q", dm.div_ceil(QK));
                // SAFETY: job j owns its expert's gate, up and down slices of gu and dd, its hidden of hh, and its
                // blocks of qb (which is this thread's own)
                unsafe {
                    let (gj, uj) = (gp.slice(g0 + 2 * j * width, width), gp.slice(g0 + (2 * j + 1) * width, width));
                    kops::q8k_quantize(x, qb.sub_mut(0..dm / QK));
                    mats[j][0].apply(qb.sub(0..dm / QK), gj);
                    mats[j][1].apply(qb.sub(0..dm / QK), uj);
                    let hj = hp.slice(h0 + j * width, width);
                    for r in 0..width {
                        hj[r] = situ(gj[r], uj[r], beta, lb);
                    }
                    kops::q8k_quantize(hj, qb.sub_mut(0..width / QK));
                    mats[j][2].apply(qb.sub(0..width / QK), dp.slice(d0 + j * dm, dm));
                }
            });
        }
        // the routing weights, in increasing expert id (the order the per-token path added them)
        for (j, &i) in idx.iter().enumerate() {
            let weight = probs[chosen[i].0 as usize] / sum * self.h.route_scale;
            for r in 0..dm {
                out[r] += weight * dd[j * dm + r];
            }
        }
        Ok(out)
    }

    /// One expert's whole FFN on the CPU, added to `out` with its routing weight: gate and up over the Q8 input, the
    /// SiTU, then down over the quantised hidden. The g, u, h and d buffers are the caller's.
    #[allow(clippy::too_many_arguments)]
    fn expert_cpu(
        &self,
        l: u32,
        e: usize,
        weight: f32,
        x: &[f32],
        qbuf: &mut Q8Buf,
        out: &mut [f32],
        g: &mut [f32],
        u: &mut [f32],
        h: &mut [f32],
        d: &mut [f32],
    ) -> Result<()> {
        let dm = self.h.n_embd as usize;
        let width = g.len();
        let group = self.groups[l as usize];
        let packs = self.w.experts_of(l, group, &[e as u32])?;
        let (data, pack) = packs[0];
        let gp = &pack.gate;
        let up = &pack.up;
        let dp = &pack.down;
        let gd = &data[gp.at as usize..(gp.at + gp.bytes) as usize];
        let ud = &data[up.at as usize..(up.at + up.bytes) as usize];
        let dd = &data[dp.at as usize..(dp.at + dp.bytes) as usize];
        let mg = Mat { ty: gp.ty, rows: gp.rows as usize, cols: gp.cols as usize, data: gd };
        let mu = Mat { ty: up.ty, rows: up.rows as usize, cols: up.cols as usize, data: ud };
        let md = Mat { ty: dp.ty, rows: dp.rows as usize, cols: dp.cols as usize, data: dd };
        let q = qbuf.quant(x);
        mg.apply(q, g);
        mu.apply(q, u);
        for i in 0..width {
            h[i] = situ(g[i], u[i], self.h.situ_beta, self.h.situ_linear_beta);
        }
        let qh = qbuf.quant(h);
        md.apply(qh, d);
        for i in 0..dm {
            out[i] += weight * d[i];
        }
        Ok(())
    }

    /// The NPU tier's share of a token's route, with the rest on the CPU. The hot experts' gate and up products go up
    /// as one graph while the CPU runs the experts it keeps; their SiTU is the CPU's, then their down products go up as
    /// a second graph. Their outputs are added after the CPU experts', in increasing expert id.
    ///
    /// The tier is not bit-exact (f32 activations, the HTP's own sums), so this only runs with TORC_NPU=1: with it off
    /// [`crate::npu::hot_of`] returns None for every expert and the CPU path is the only one taken.
    #[allow(clippy::too_many_arguments)]
    fn moe_staged(
        &self,
        l: u32,
        x: &[f32],
        qbuf: &mut Q8Buf,
        chosen: &[(u32, f32)],
        probs: &[f32],
        sum: f32,
        hot: &[Option<[i32; 3]>],
    ) -> Result<Vec<f32>> {
        let dm = self.h.n_embd as usize;
        let group = self.groups[l as usize];
        let width = self.w.layer_experts(group).width as usize;
        // (index in chosen) of the experts the NPU holds
        let npu: Vec<usize> = (0..chosen.len()).filter(|&i| hot[i].is_some()).collect();
        let mut gu = crate::scratch::buf::<f32>("inkling.moe_gu", npu.len() * 2 * width);
        let mut hh = crate::scratch::buf::<f32>("inkling.moe_h", npu.len() * width);
        let mut dd = crate::scratch::buf::<f32>("inkling.moe_d", npu.len() * dm);
        let ups: Vec<crate::npu::GateUp> = npu
            .iter()
            .enumerate()
            .map(|(j, &i)| {
                let ids = hot[i].expect("a hot expert's ids");
                crate::npu::GateUp {
                    ids: [ids[0], ids[1]],
                    x: x.as_ptr(),
                    n: 1,
                    g: gu[j * 2 * width..].as_mut_ptr(),
                    u: gu[(j * 2 + 1) * width..].as_mut_ptr(),
                }
            })
            .collect();
        // SAFETY: x holds dm floats, which begin copies in; gu holds each expert's gate and up rows until finish
        let started = unsafe { crate::npu::begin_gate_up(&ups, dm) };
        let skip: Vec<bool> = hot.iter().map(Option::is_some).collect();
        let cpu = self.moe_cpu(l, x, qbuf, chosen, probs, sum, &skip);
        // the graph is waited for before anything else, so a failed CPU expert leaves nothing in flight
        let mut down = started && crate::npu::finish();
        let mut out = cpu?;
        if down {
            for (j, _) in npu.iter().enumerate() {
                let (g, u) = (&gu[j * 2 * width..j * 2 * width + width], &gu[(j * 2 + 1) * width..(j * 2 + 2) * width]);
                for r in 0..width {
                    hh[j * width + r] = situ(g[r], u[r], self.h.situ_beta, self.h.situ_linear_beta);
                }
            }
            let downs: Vec<crate::npu::Down> = npu
                .iter()
                .enumerate()
                .map(|(j, &i)| crate::npu::Down {
                    id: hot[i].expect("a hot expert's ids")[2],
                    x: hh[j * width..].as_ptr(),
                    n: 1,
                    d: dd[j * dm..].as_mut_ptr(),
                })
                .collect();
            // SAFETY: hh holds each expert's hidden rows (begin copies them in), and dd its output rows, until finish
            down = unsafe { crate::npu::begin_down(&downs, width) } && crate::npu::finish();
        }
        if down {
            for (j, &i) in npu.iter().enumerate() {
                let weight = probs[chosen[i].0 as usize] / sum * self.h.route_scale;
                for r in 0..dm {
                    out[r] += weight * dd[j * dm + r];
                }
            }
            return Ok(out);
        }
        // the NPU did not take them after all: their own CPU chain, after the experts it kept
        let mut g = vec![0.0f32; width];
        let mut u = vec![0.0f32; width];
        let mut h = vec![0.0f32; width];
        let mut d = vec![0.0f32; dm];
        for &i in &npu {
            let e = chosen[i].0 as usize;
            let weight = probs[e] / sum * self.h.route_scale;
            self.expert_cpu(l, e, weight, x, qbuf, &mut out, &mut g, &mut u, &mut h, &mut d)?;
        }
        Ok(out)
    }

    fn head(&self, x: &[f32], logits: &mut [f32], qbuf: &mut Q8Buf) -> Result<()> {
        let t = self.w.head()?;
        let (_, rows) = t.shape();
        let rb = t.row_bytes();
        let rows = rows.min(self.h.n_vocab_unpadded as usize);
        if logits.len() != rows {
            return err(format!("logits must hold {rows} rows, not {}", logits.len()));
        }
        let m = Mat { ty: t.t.ty, rows, cols: self.h.n_embd as usize, data: &t.data[..rows * rb] };
        let q = qbuf.quant(x);
        m.apply(q, logits);
        Ok(())
    }
}

impl LayerW {
    fn load(w: &Weights, h: &Hp, l: u32) -> Result<LayerW> {
        let dm = h.n_embd as usize;
        let hd = h.head_dim as usize;
        let nh = h.n_head as usize;
        let nkv = h.n_head_kv as usize;
        let k = h.n_conv as usize;
        let rel = h.rel_len as usize;
        let is_swa = h.swa[l as usize];
        let f = |s: &str| to_f32(&w.layer(l, s)?);
        let (attn_norm, q_norm, k_norm, v_norm) = if is_swa {
            let n = f("attn_norm.weight")?;
            (n.clone(), n.clone(), n.clone(), n)
        } else {
            (vec![], f("attn_q_inp_norm.weight")?, f("attn_k_inp_norm.weight")?, f("attn_v_inp_norm.weight")?)
        };
        let ffn_norm = f("ffn_norm.weight")?;
        let q_clip = f("attn_q_clip.weight")?;
        let rel_clip = f("attn_rel_clip.weight")?;
        let rel_bias = f("attn_rel_bias.weight")?;
        let sink_proj = f("attn_sink_proj.weight")?;
        let k_conv = f("attn_k_conv.weight")?;
        let v_conv = f("attn_v_conv.weight")?;
        let a_conv = f("attn_post_conv.weight")?;
        let f_conv = f("ffn_post_conv.weight")?;
        let attn_res_query = if l > 0 { f("attn_res_query.weight")? } else { vec![] };
        let ffn_res_query = f("ffn_res_query.weight")?;
        let router = f("ffn_gate_inp.weight")?;
        let exp_probs_b = match w.tensor(&format!("blk.{l}.exp_probs_b.bias")) {
            Some(t) => t.f32s()?,
            None => vec![0.0; h.n_expert as usize],
        };
        let loop_ctrl = if (h.core_start..h.core_end).contains(&l) {
            let msg_query = f("loop_msg_query.weight")?;
            let r = h.rank as usize;
            let mut down = Vec::with_capacity(4 * r * 2 * dm);
            let mut gain = Vec::with_capacity(4 * dm * r);
            let mut forget = Vec::with_capacity(4 * dm * r);
            for c in ["q", "k", "v", "ffn"] {
                down.extend(f(&format!("loop_{c}_ctrl_down.weight"))?);
                gain.extend(f(&format!("loop_{c}_ctrl_gain.weight"))?);
                forget.extend(f(&format!("loop_{c}_ctrl_forget.weight"))?);
            }
            Some(LoopW { msg_query, down, gain, forget })
        } else {
            None
        };
        // the small tensors' lengths are the forward pass's own checks
        if ffn_norm.len() != dm
            || q_clip.len() != nh * hd
            || rel_clip.len() != nh * rel
            || rel_bias.len() != h.rel_extent as usize * rel
            || sink_proj.len() != nh * dm
            || k_conv.len() != nkv * hd * k
            || v_conv.len() != nkv * hd * k
            || a_conv.len() != dm * k
            || f_conv.len() != dm * k
            || ffn_res_query.len() != dm
            || router.len() != h.n_expert as usize * dm
            || exp_probs_b.len() != h.n_expert as usize
        {
            return err(format!("blk.{l}: a resident tensor has the wrong shape"));
        }
        if !is_swa && (q_norm.len() != dm || k_norm.len() != dm || v_norm.len() != dm) {
            return err(format!("blk.{l}: an attention input norm has the wrong width"));
        }
        if is_swa && attn_norm.len() != dm {
            return err(format!("blk.{l}: attn_norm has the wrong width"));
        }
        if l > 0 && attn_res_query.len() != (if is_swa { 1 } else { 3 }) * dm {
            return err(format!("blk.{l}: attn_res_query has the wrong shape"));
        }
        Ok(LayerW {
            attn_norm,
            q_norm,
            k_norm,
            v_norm,
            ffn_norm,
            q_clip,
            rel_clip,
            rel_bias,
            sink_proj,
            k_conv,
            v_conv,
            a_conv,
            f_conv,
            attn_res_query,
            ffn_res_query,
            router,
            exp_probs_b,
            loop_ctrl,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use format::write::{plan, write};
    use format::{Directory, Experts, Origin, Pack, Piece, Tensor, page_up};

    fn kv(out: &mut Vec<u8>, k: &str, t: u32, v: &[u8]) {
        out.extend((k.len() as u64).to_le_bytes());
        out.extend(k.as_bytes());
        out.extend(t.to_le_bytes());
        out.extend(v);
    }

    fn meta() -> Vec<u8> {
        let mut e: Vec<(String, u32, Vec<u8>)> = Vec::new();
        let s = |x: &str| {
            let mut v = (x.len() as u64).to_le_bytes().to_vec();
            v.extend(x.as_bytes());
            v
        };
        e.push(("general.architecture".into(), 8, s("inkling2_loop")));
        let a = "inkling2_loop.";
        for (k, v) in [
            ("block_count", 3u32),
            ("embedding_length", 256),
            ("attention.head_count", 2),
            ("attention.head_count_kv", 1),
            ("attention.key_length", 128),
            ("attention.value_length", 128),
            ("expert_count", 2),
            ("expert_used_count", 1),
            ("expert_feed_forward_length", 256),
            ("vocab_size", 8),
            ("vocab_size_unpadded", 8),
            ("context_length", 16),
            ("shortconv.l_cache", 4),
            ("attention.sliding_window", 8),
            ("attention.relative_length", 2),
            ("attention.relative_extent", 8),
            ("attention.log_scale_floor", 8192),
            ("attn_res.block_size", 1),
            ("loop.start", 1),
            ("loop.end", 2),
            ("loop.count", 2),
            ("loop.gate_rank", 2),
        ] {
            e.push((format!("{a}{k}"), 4, v.to_le_bytes().to_vec()));
        }
        for (k, v) in [
            ("attention.layer_norm_rms_epsilon", 1e-6f32),
            ("attention.qk_clip", 22.627417),
            ("attention.relative_clip", 8.0),
            ("attention.log_scale_alpha", 0.1),
            ("final_logit_softcapping", 30.0),
            ("activation.situ_beta", 4.0),
            ("activation.situ_linear_beta", 25.0),
            ("expert_weights_scale", 8.0),
        ] {
            e.push((format!("{a}{k}"), 6, v.to_le_bytes().to_vec()));
        }
        let mut arr = 7u32.to_le_bytes().to_vec();
        arr.extend(3u64.to_le_bytes());
        arr.extend([1u8, 1, 0]);
        e.push((format!("{a}attention.sliding_window_pattern"), 9, arr));
        let mut b = (e.len() as u64).to_le_bytes().to_vec();
        for (k, t, v) in e {
            kv(&mut b, &k, t, &v);
        }
        b
    }

    fn tensor(name: &str, ty: u32, dims: &[u64]) -> Tensor {
        let mut d = [1u64; 4];
        d[..dims.len()].copy_from_slice(dims);
        let bytes = dims[1..].iter().fold(dims[0], |a, &x| a * x);
        Tensor {
            name: name.into(),
            ty,
            n_dims: dims.len() as u32,
            dims: d,
            bytes: format::matrix_bytes(ty, bytes / dims[0], dims[0]).unwrap(),
            ..Default::default()
        }
    }

    fn piece(name: &str, ty: u32, rows: u32, cols: u32, at: u64) -> Piece {
        Piece {
            name: name.into(),
            ty,
            rows,
            cols,
            at,
            bytes: format::matrix_bytes(ty, rows as u64, cols as u64).unwrap(),
            origin: Origin::default(),
            stride: 0,
        }
    }

    /// A tiny three-layer looped model (prelude 0, core 1 visited twice, postlude 2) written as a ToRc file and run:
    /// every tensor is zero-filled, so the check is that every shape and the loop/caches line up and the logits are
    /// finite.
    #[test]
    fn tiny_forward() {
        let dir = std::env::temp_dir().join(format!("torc-inkling-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tiny.torc");
        let mut ts = vec![
            tensor("token_embd.weight", 8, &[256, 8]),
            tensor("output.weight", 8, &[256, 8]),
            tensor("token_embd_norm.weight", 0, &[256]),
            tensor("output_norm.weight", 0, &[256]),
            tensor("output_res_query.weight", 0, &[256]),
        ];
        for l in 0..3u32 {
            let q = if l == 2 { 3 } else { 1 };
            ts.extend([
                tensor(&format!("blk.{l}.attn_q.weight"), 8, &[256, 256]),
                tensor(&format!("blk.{l}.attn_k.weight"), 8, &[256, 128]),
                tensor(&format!("blk.{l}.attn_v.weight"), 8, &[256, 128]),
                tensor(&format!("blk.{l}.attn_output.weight"), 8, &[256, 256]),
                tensor(&format!("blk.{l}.attn_gate.weight"), 8, &[256, 256]),
                tensor(&format!("blk.{l}.attn_rel.weight"), 8, &[256, 4]),
                tensor(&format!("blk.{l}.attn_q_clip.weight"), 0, &[128, 2]),
                tensor(&format!("blk.{l}.attn_rel_clip.weight"), 0, &[2, 2]),
                tensor(&format!("blk.{l}.attn_rel_bias.weight"), 0, &[2, 8]),
                tensor(&format!("blk.{l}.attn_sink_proj.weight"), 0, &[256, 2]),
                tensor(&format!("blk.{l}.attn_k_conv.weight"), 0, &[4, 128]),
                tensor(&format!("blk.{l}.attn_v_conv.weight"), 0, &[4, 128]),
                tensor(&format!("blk.{l}.attn_post_conv.weight"), 0, &[4, 256]),
                tensor(&format!("blk.{l}.ffn_post_conv.weight"), 0, &[4, 256]),
                tensor(&format!("blk.{l}.ffn_norm.weight"), 0, &[256]),
                tensor(&format!("blk.{l}.ffn_gate_inp.weight"), 0, &[256, 2]),
                tensor(&format!("blk.{l}.exp_probs_b.bias"), 0, &[2]),
                tensor(&format!("blk.{l}.ffn_res_query.weight"), 0, &[256]),
            ]);
            if l > 0 {
                ts.push(tensor(&format!("blk.{l}.attn_res_query.weight"), 0, &[256, q]));
            }
            if l == 1 {
                ts.push(tensor(&format!("blk.{l}.loop_msg_query.weight"), 0, &[256, 4]));
                for c in ["q", "k", "v", "ffn"] {
                    ts.push(tensor(&format!("blk.{l}.loop_{c}_ctrl_down.weight"), 0, &[512, 2]));
                    ts.push(tensor(&format!("blk.{l}.loop_{c}_ctrl_gain.weight"), 0, &[2, 256]));
                    ts.push(tensor(&format!("blk.{l}.loop_{c}_ctrl_forget.weight"), 0, &[2, 256]));
                }
            }
            if l != 2 {
                ts.push(tensor(&format!("blk.{l}.attn_norm.weight"), 0, &[256]));
            } else {
                for n in ["attn_q_inp_norm", "attn_k_inp_norm", "attn_v_inp_norm"] {
                    ts.push(tensor(&format!("blk.{l}.{n}.weight"), 0, &[256]));
                }
                ts.push(tensor(&format!("blk.{l}.attn_k_conv_proj.weight"), 8, &[256, 4]));
                ts.push(tensor(&format!("blk.{l}.attn_v_conv_proj.weight"), 8, &[256, 4]));
            }
        }
        let mut experts = Vec::new();
        for layer in 0..3u32 {
            let g = piece("g", 8, 256, 256, 0);
            let mut up = piece("u", 8, 256, 256, page_up(g.bytes));
            let mut d = piece("d", 8, 256, 256, page_up(up.at + up.bytes));
            up.at = page_up(g.bytes);
            d.at = page_up(up.at + up.bytes);
            let bytes = page_up(d.at + d.bytes);
            experts.push(Experts {
                layer,
                n_expert: 2,
                n_used: 1,
                d_in: 256,
                width: 256,
                first: 0,
                pack: Pack { bytes, gate: g, up, down: d },
                crcs: Vec::new(),
            });
        }
        let d = Directory { metadata: meta(), source: None, tensors: ts, experts };
        let p = plan(d, "inkling2_loop").unwrap();
        let h = write(path.to_str().unwrap(), p, |_, _, _| Ok(()), |_, _, _| Ok(())).unwrap();
        assert!(h.file_bytes > 0);
        let mut m = Inkling::load(path.to_str().unwrap(), 16).unwrap();
        let mut l = vec![0.0f32; 8];
        m.forward(0, 0, &mut l).unwrap();
        assert!(l.iter().all(|v| v.is_finite()));
        m.forward(1, 1, &mut l).unwrap();
        assert!(l.iter().all(|v| v.is_finite()));
        std::fs::remove_dir_all(&dir).ok();
    }
}
