//! torc-model - the model's forward pass over a ToRc file.
//!
//! Token ids go in and logits come out. Each layer normalises and quantises its input, runs the mixer (gated DeltaNet
//! or attention), normalises and quantises again and runs the experts; after the last layer come the output norm and
//! the vocabulary product. The Qwen family and its relatives run on Model (models/qwen), Gemma 4 and gpt-oss on their
//! own (models/gemma, models/gptoss), on the CPU, with the NPU, a GPU and a second machine as optional tiers on top.
//!
//! The code is in five folders: layers/ (the building blocks every model shares), models/ (one folder per model
//! family), runtime/ (weights, expert cache, compute pool, two machines), tiers/ (the NPU and the GPU) and chat/
//! (tokenizers and the chat APIs).
//!
//! The CPU path is deterministic: a row gives the same bits whichever thread computes it, so every run repeats exactly.

pub mod chat;
pub mod layers;
pub mod models;
pub mod runtime;
pub mod tiers;

// every module stays reachable at the crate root by its own name (crate::attn, model::gemma, ...)
pub(crate) use chat::uniflags;
pub use chat::{api, serve, tok};
pub use kernels::switches;
pub use layers::{act, attn, fattn, mla, moe, ops, rec};
#[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), any(windows, target_os = "linux")))]
pub use models::gemma::npu_gemma;
pub use models::gemma::{gemma, gemma_mtp};
pub use models::gptoss::gptoss;
pub use models::inkling2::model::Inkling;
pub use models::qwen::hparams::Hparams;
pub use models::qwen::{forward, hc, hparams, mtp, mtp_hc};
pub use runtime::{
    ecache, ecache_own, ep, markfile, pipe, plan, pool, preflight, profile, repack, scratch, tune, weights,
};
#[cfg(feature = "gpu")]
pub use tiers::gpu;
pub use tiers::npu;

// Model is the forward pass the Qwen family shares with its relatives (ERNIE 4.5, GLM-4.7-Flash, Moonlight, LFM2), and
// most of the tools take it; it sits in models/qwen because Qwen is the family it was written for.
pub use forward::{Mark, Model};

use std::sync::atomic::{AtomicU64, Ordering};

/// Time per phase of a block function, summed over all calls into a static table when TORC_PROF=1 (torc-run prints the
/// tables). Use it as `let mut l = Laps::start(&TABLE); ...; l.lap(i);`, which adds the time since the last lap to
/// phase i.
pub struct Laps(&'static [AtomicU64], Option<std::time::Instant>);

impl Laps {
    pub fn start(ns: &'static [AtomicU64]) -> Laps {
        static ON: std::sync::LazyLock<bool> =
            std::sync::LazyLock::new(|| crate::switches::get("TORC_PROF").is_some_and(|v| v != "0"));
        Laps(ns, if *ON { Some(std::time::Instant::now()) } else { None })
    }

    /// Whether TORC_PROF is on, i.e. whether this run collects phase times.
    pub fn on(&self) -> bool {
        self.1.is_some()
    }

    pub fn lap(&mut self, i: usize) {
        if let Some(t) = &mut self.1 {
            let now = std::time::Instant::now();
            self.0[i].fetch_add((now - *t).as_nanos() as u64, Ordering::Relaxed);
            *t = now;
        }
    }
}

/// One phase table: its name, the names of its phases and each phase's nanoseconds.
pub type PhaseTable = (&'static str, &'static [&'static str], &'static [AtomicU64]);

/// The phase tables.
pub fn phase_tables() -> [PhaseTable; 5] {
    [
        ("moe", &moe::MOE_PHASES, &moe::MOE_NS),
        ("moe 1 token", &moe::MOE1_PHASES, &moe::MOE1_NS),
        ("attention", &attn::ATTN_PHASES, &attn::ATTN_NS),
        ("deltanet", &rec::REC_PHASES, &rec::REC_NS),
        ("mixers", &hc::HC_PHASES, &hc::HC_NS),
    ]
}

#[cfg(windows)]
unsafe extern "system" {
    fn GetCurrentThread() -> *mut core::ffi::c_void;
    fn SetThreadAffinityMask(h: *mut core::ffi::c_void, mask: usize) -> usize;
}

/// First thing in each program's main, while it has one thread: points the NPU's DSP loader at its libraries
/// (ADSP_LIBRARY_PATH, see ggml::set_dsp_path). Does nothing where there is no NPU tier.
pub fn set_npu_env() {
    #[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), any(windows, target_os = "linux")))]
    kernels::ggml::set_dsp_path();
}

/// Starts the compute pool: one worker per core, each pinned to its core. Left unpinned, Windows moves the workers
/// around and the big products lose about a quarter of their bandwidth. TORC_PIN=0 leaves them unpinned. Off Windows
/// they are left to the OS unless TORC_PIN=1 (as pool::init).
pub fn init_threads(n: usize) {
    let pin = crate::switches::get("TORC_PIN").map(|v| v != "0").unwrap_or(cfg!(windows));
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .start_handler(move |i| {
            #[cfg(windows)]
            if pin {
                unsafe {
                    SetThreadAffinityMask(GetCurrentThread(), 1usize << (i % usize::BITS as usize));
                }
            }
            #[cfg(target_os = "linux")]
            if pin {
                crate::pool::pin_linux(i as i32);
            }
            #[cfg(target_os = "macos")]
            crate::pool::user_initiated();
            let _ = (i, pin);
        })
        .build_global();
}
