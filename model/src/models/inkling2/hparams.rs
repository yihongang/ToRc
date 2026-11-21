//! hparams.rs - Inkling 2's hyperparameters, from the file's GGUF metadata.
//!
//! The model runs a prelude of layers, a looped core that is visited `loop.count` times, and a postlude. Every layer
//! visit owns its own KV cache and short-convolution state; the file stores only the physical layers, so the number of
//! visits is the number of cache slots the forward pass needs.

use format::meta::{Meta, Value};
use format::read::Model as File;
use format::{Result, err};

#[derive(Clone, Debug, PartialEq)]
pub struct Hp {
    pub arch: String,
    /// physical layers in the file
    pub n_layer: u32,
    /// layer visits, i.e. cache slots: the physical layers plus the core's extra passes
    pub n_slot: u32,
    pub n_embd: u32,
    pub n_head: u32,
    pub n_head_kv: u32,
    pub head_dim: u32,
    pub n_expert: u32,
    pub n_used: u32,
    pub d_ff: u32,
    pub n_vocab: u32,
    pub n_vocab_unpadded: u32,
    pub n_ctx: u32,
    pub eps: f32,
    pub n_conv: u32,
    /// whether each physical layer is a sliding-window layer
    pub swa: Vec<bool>,
    pub window: u32,
    pub rel_len: u32,
    pub rel_extent: u32,
    pub qk_clip: f32,
    pub rel_clip: f32,
    pub log_floor: u32,
    pub log_alpha: f32,
    pub block: u32,
    pub softcap: f32,
    pub situ_beta: f32,
    pub situ_linear_beta: f32,
    pub route_scale: f32,
    pub core_start: u32,
    pub core_end: u32,
    pub passes: u32,
    pub rank: u32,
}

fn first_u32(m: &Meta, key: &str) -> Option<u32> {
    match m.get(&m.arch_key(key))? {
        Value::Int(i) => u32::try_from(*i).ok(),
        Value::Array(_, v) => match v.first()? {
            Value::Int(i) => u32::try_from(*i).ok(),
            _ => None,
        },
        _ => None,
    }
}

fn bools(m: &Meta, key: &str) -> Option<Vec<bool>> {
    match m.get(&m.arch_key(key))? {
        Value::Array(_, v) => v
            .iter()
            .map(|x| match x {
                Value::Bool(b) => Some(*b),
                Value::Int(i) => Some(*i != 0),
                _ => None,
            })
            .collect(),
        _ => None,
    }
}

impl Hp {
    pub fn from_file(f: &File) -> Result<Hp> {
        let m = Meta::parse(&f.metadata)?;
        let arch = m.str("general.architecture").unwrap_or("").to_string();
        if arch != "inkling2_loop" {
            return err(format!("architecture {arch} is not inkling2_loop"));
        }
        let need = |k: &str| m.arch_u32(k).ok_or_else(|| format::Error(format!("missing {arch}.{k}")));
        let needf = |k: &str| m.arch_f32(k).ok_or_else(|| format::Error(format!("missing {arch}.{k}")));
        let n_layer = need("block_count")?;
        let n_embd = need("embedding_length")?;
        let n_head = first_u32(&m, "attention.head_count").ok_or_else(|| format::Error("missing head_count".into()))?;
        let n_head_kv =
            first_u32(&m, "attention.head_count_kv").ok_or_else(|| format::Error("missing head_count_kv".into()))?;
        let head_dim = m
            .arch_u32("attention.key_length")
            .or_else(|| m.arch_u32("attention.value_length"))
            .ok_or_else(|| format::Error("missing attention.key_length".into()))?;
        let n_expert = need("expert_count")?;
        let n_used = need("expert_used_count")?;
        let d_ff = need("expert_feed_forward_length")?;
        let n_vocab = need("vocab_size")?;
        let n_vocab_unpadded = need("vocab_size_unpadded")?;
        let n_ctx = m.arch_u32("context_length").unwrap_or(0);
        let eps = needf("attention.layer_norm_rms_epsilon")?;
        let n_conv = need("shortconv.l_cache")?;
        let swa = bools(&m, "attention.sliding_window_pattern")
            .ok_or_else(|| format::Error("missing attention.sliding_window_pattern".into()))?;
        let window = need("attention.sliding_window")?;
        let rel_len = need("attention.relative_length")?;
        let rel_extent = need("attention.relative_extent")?;
        let qk_clip = needf("attention.qk_clip")?;
        let rel_clip = needf("attention.relative_clip")?;
        let log_floor = need("attention.log_scale_floor")?;
        let log_alpha = needf("attention.log_scale_alpha")?;
        let block = need("attn_res.block_size")?;
        let softcap = m.arch_f32("final_logit_softcapping").unwrap_or(0.0);
        let situ_beta = needf("activation.situ_beta")?;
        let situ_linear_beta = needf("activation.situ_linear_beta")?;
        let route_scale = m.arch_f32("expert_weights_scale").unwrap_or(1.0);
        // the loop keys are named loop.start / loop.end / loop.count / loop.gate_rank in the mainline file
        let core_start = need("loop.start")?;
        let core_end = need("loop.end")?;
        let passes = need("loop.count")?;
        let rank = need("loop.gate_rank")?;

        if n_layer == 0
            || n_embd == 0
            || n_head == 0
            || n_head_kv == 0
            || !n_head.is_multiple_of(n_head_kv)
            || head_dim == 0
            || n_expert == 0
            || n_used == 0
            || n_used > n_expert
            || d_ff == 0
            || n_vocab_unpadded == 0
            || n_vocab_unpadded > n_vocab
            || eps <= 0.0
            || !(2..=64).contains(&n_conv)
            || swa.len() != n_layer as usize
            || window == 0
            || rel_len == 0
            || rel_extent == 0
            || qk_clip <= 0.0
            || rel_clip <= 0.0
            || log_floor == 0
            || log_alpha < 0.0
            || block == 0
            || situ_beta <= 0.0
            || situ_linear_beta <= 0.0
        {
            return err("inkling2_loop: invalid dimensions or hyperparameters");
        }
        if passes < 2
            || core_start == 0
            || core_start >= core_end
            || core_end > n_layer
            || !core_start.is_multiple_of(block)
            || !core_end.is_multiple_of(block)
            || rank == 0
        {
            return err(format!(
                "inkling2_loop: invalid loop core [{core_start}, {core_end}) with block size {block}, {passes} passes or gate rank {rank}"
            ));
        }
        let n_slot = n_layer + (passes - 1) * (core_end - core_start);
        Ok(Hp {
            arch,
            n_layer,
            n_slot,
            n_embd,
            n_head,
            n_head_kv,
            head_dim,
            n_expert,
            n_used,
            d_ff,
            n_vocab,
            n_vocab_unpadded,
            n_ctx,
            eps,
            n_conv,
            swa,
            window,
            rel_len,
            rel_extent,
            qk_clip,
            rel_clip,
            log_floor,
            log_alpha,
            block,
            softcap,
            situ_beta,
            situ_linear_beta,
            route_scale,
            core_start,
            core_end,
            passes,
            rank,
        })
    }

    /// The physical layer a visit reads its weights from: visits past the physical layers cover the core again.
    pub fn layer_of(&self, visit: u32) -> u32 {
        if visit < self.n_layer {
            visit
        } else {
            self.core_start + (visit - self.n_layer) % (self.core_end - self.core_start)
        }
    }

    /// The virtual layer index a visit runs as: `ip` in the reference's `build_layer(il, ip, ...)`. For the first pass
    /// it is the physical layer; later core passes restart at `core_start`.
    pub fn virtual_of(&self, visit: u32) -> u32 {
        if visit < self.n_layer {
            visit
        } else {
            self.core_start + (visit - self.n_layer) % (self.core_end - self.core_start)
        }
    }

    /// The layer visits, in execution order, as (visit slot, virtual index, pass).
    pub fn schedule(&self) -> Vec<(u32, u32, u32)> {
        let mut out = Vec::with_capacity(self.n_slot as usize);
        for i in 0..self.core_start {
            out.push((i, i, 0));
        }
        for p in 0..self.passes {
            for ip in self.core_start..self.core_end {
                let visit = if p == 0 {
                    ip
                } else {
                    self.n_layer + (p - 1) * (self.core_end - self.core_start) + (ip - self.core_start)
                };
                out.push((visit, ip, p));
            }
        }
        for i in self.core_end..self.n_layer {
            out.push((i, i, 0));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The schedule has one visit per slot, the prelude and postlude once, the core `passes` times.
    #[test]
    fn schedule_visits() {
        let h = Hp {
            arch: "inkling2_loop".into(),
            n_layer: 24,
            n_slot: 48,
            n_embd: 2304,
            n_head: 24,
            n_head_kv: 8,
            head_dim: 128,
            n_expert: 128,
            n_used: 8,
            d_ff: 1152,
            n_vocab: 201024,
            n_vocab_unpadded: 200058,
            n_ctx: 1 << 20,
            eps: 1e-6,
            n_conv: 4,
            swa: vec![true; 24],
            window: 1024,
            rel_len: 16,
            rel_extent: 1024,
            qk_clip: 22.627417,
            rel_clip: 8.0,
            log_floor: 8192,
            log_alpha: 0.1,
            block: 6,
            softcap: 30.0,
            situ_beta: 4.0,
            situ_linear_beta: 25.0,
            route_scale: 8.0,
            core_start: 6,
            core_end: 18,
            passes: 3,
            rank: 64,
        };
        let s = h.schedule();
        assert_eq!(s.len(), 48);
        assert_eq!(s[0], (0, 0, 0));
        assert_eq!(s[6], (6, 6, 0));
        assert_eq!(s[18], (24, 6, 1));
        assert_eq!(s[30], (36, 6, 2));
        assert_eq!(s[42], (18, 18, 0));
        assert_eq!(s[47], (23, 23, 0));
        assert_eq!(h.layer_of(24), 6);
        assert_eq!(h.layer_of(47), 17);
    }
}
