//! preflight.rs - "will this run": a GGUF (before it is converted) or a ToRc file checked against what this build can
//! run, with the checks the load itself makes: the architecture, every weight's type, the model's hyperparameters and
//! each layer's experts. So a model that would stop at load stops here instead, in a second or two and by name,
//! before the time and the disk space go on converting it. `torc plan <file>` prints it; `torc convert-gguf` and the
//! app ask it first.

use crate::models;
use format::read::Model as File;
use format::{Result, err};

/// What the check found about a model that can run.
pub struct Preflight {
    pub arch: String,
    /// the ToRc file's size (a GGUF converts to about the same)
    pub file_bytes: u64,
    pub layers: u32,
    /// the machine's installed memory, where it is known
    pub ram: Option<u64>,
}

impl Preflight {
    /// One line on how the model fits this machine's memory.
    pub fn memory_note(&self) -> String {
        let gb = |b: u64| b as f64 / (1u64 << 30) as f64;
        match self.ram {
            Some(r) if self.file_bytes + (4u64 << 30) <= r => {
                format!("{:.1} GB, fits in this machine's {:.0} GB of memory", gb(self.file_bytes), gb(r))
            }
            Some(r) => format!(
                "{:.1} GB, more than this machine's {:.0} GB of memory holds with room to spare: the experts that \
                 don't fit are read from the disk as they are needed, which is slower (EXPERT-CACHE.md)",
                gb(self.file_bytes),
                gb(r)
            ),
            None => format!("{:.1} GB", gb(self.file_bytes)),
        }
    }
}

/// Checks a model file: a GGUF by its name (.gguf, or the first part of a split one), else a ToRc file.
pub fn check_path(path: &str) -> Result<Preflight> {
    let lower = path.to_ascii_lowercase();
    let f = if lower.ends_with(".gguf") || lower.ends_with(".gguf_file") {
        format::convert::plan_gguf(path)?
    } else {
        File::open(path)?
    };
    check(&f)
}

/// Whether layer `l` has routed experts, found as the loads find them (a dense model's FFN is its layer's one expert).
fn has_experts(f: &File, l: u32) -> bool {
    f.blocks.iter().any(|b| b.layer == l && b.experts.is_some())
}

/// The checks, on a file as read (or as a GGUF would become, format::convert::plan_gguf).
pub fn check(f: &File) -> Result<Preflight> {
    let arch = f.header.arch.clone();
    if !models::ARCHS.contains(&arch.as_str()) {
        if arch == "inkling2" {
            return err(
                "its architecture, inkling2, is the Unsloth runtime's loop layout: ToRc runs the Mainline_Support \
                 files, whose architecture is inkling2_loop (use Mainline_Support/<model>.gguf_file)",
            );
        }
        let named = if arch.is_empty() { "unnamed" } else { arch.as_str() };
        return err(format!(
            "its architecture, {named}, isn't one ToRc runs yet (it runs {})",
            models::ARCHS.join(", ")
        ));
    }
    if let Some((name, ty)) = crate::weights::unsupported_weight(f) {
        return err(crate::weights::unsupported_message(&name, ty));
    }
    // the hyperparameters, read and checked by the model's own code, then each layer's experts, which every load needs
    let layers = if models::QWEN_ENGINE.contains(&arch.as_str()) {
        models::qwen::hparams::Hparams::from_file(f)?.n_layer
    } else if arch == "gemma4" {
        models::gemma::gemma::GemmaHp::from_meta(&f.metadata)?.n_layer as u32
    } else if arch == "inkling2_loop" {
        models::inkling2::hparams::Hp::from_file(f)?.n_layer
    } else {
        models::gptoss::gptoss::OssHp::from_meta(&f.metadata)?.n_layer as u32
    };
    if let Some(l) = (0..layers).find(|&l| !has_experts(f, l)) {
        let which =
            if arch == "gemma4" { " (of Gemma 4, ToRc runs the 26B-A4B, not the dense models yet)" } else { "" };
        return err(format!(
            "layer {l} has no routed experts: ToRc runs {arch} as a mixture-of-experts model, and this one is \
             dense{which}"
        ));
    }
    Ok(Preflight { arch, file_bytes: f.header.file_bytes, layers, ram: crate::weights::total_phys() })
}

#[cfg(test)]
mod tests {
    use format::read::Model as File;
    use format::write::plan;
    use format::*;

    /// A one-layer MoE laid out in memory as `arch`, its experts' gate of type `gate_ty`.
    fn tiny(arch: &str, gate_ty: u32) -> File {
        let piece = |name: &str, ty: u32, rows: u32, cols: u32, at| Piece {
            name: name.into(),
            ty,
            rows,
            cols,
            at,
            bytes: matrix_bytes(ty, rows as u64, cols as u64).unwrap(),
            origin: Origin::default(),
            stride: 0,
        };
        let gate = piece("blk.0.ffn_gate_exps.weight", gate_ty, 64, 256, 0);
        let up = piece("blk.0.ffn_up_exps.weight", 8, 64, 256, page_up(gate.bytes));
        let down = piece("blk.0.ffn_down_exps.weight", 8, 256, 64, page_up(up.at + up.bytes));
        let pack_bytes = page_up(down.at + down.bytes);
        let t = |name: &str, dims: [u64; 4]| Tensor {
            name: name.into(),
            ty: 0,
            n_dims: 2,
            dims,
            bytes: 4 * dims[0] * dims[1],
            ..Default::default()
        };
        let d = Directory {
            metadata: 0u64.to_le_bytes().to_vec(),
            source: None,
            tensors: vec![t("token_embd.weight", [256, 10, 1, 1]), t("blk.0.ffn_gate_inp.weight", [256, 3, 1, 1])],
            experts: vec![Experts {
                layer: 0,
                n_expert: 3,
                n_used: 2,
                d_in: 256,
                width: 64,
                first: 0,
                pack: Pack { bytes: pack_bytes, gate, up, down },
                crcs: vec![],
            }],
        };
        File::from_plan("tiny.gguf", plan(d, arch).unwrap())
    }

    /// An architecture ToRc doesn't run is refused by name before anything else is looked at, and a weight type with
    /// no kernel is refused naming the weight, as the load would refuse it.
    #[test]
    fn what_cannot_run_is_refused_by_name() {
        let e = super::check(&tiny("clip", 8)).err().expect("clip refused").0;
        assert!(e.contains("its architecture, clip, isn't one ToRc runs yet"), "{e}");
        let e = super::check(&tiny("qwen35moe", 15)).err().expect("Q8_K refused").0;
        assert!(e.contains("blk.0.ffn_gate_exps.weight is Q8_K (ggml type 15), a type ToRc has no kernel for"), "{e}");
    }
}
