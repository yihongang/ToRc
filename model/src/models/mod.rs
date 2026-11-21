//! models - one folder per model: its chat format (chat.rs: the prompt and the tool calls), and for a model with a
//! network of its own, the forward pass and its server half (serve.rs, behind chat/serve.rs's Serve). A new network is
//! a new folder here and one line in torc-server's dispatch.
//!
//! qwen: Qwen3.6 (qwen35moe, qwen35), Qwen3-Next (qwen3next), Qwen3 MoE (qwen3moe) and Qwen3.8 Flash Next (qwen4exp),
//! with their MTP blocks, and the engine the next four run on. gemma: Gemma 4 and its drafter. gptoss: gpt-oss.
//! ernie, glm, moonlight, lfm2: ERNIE 4.5, GLM-4.7-Flash, Moonlight and LFM2, whose networks the qwen engine covers,
//! so each folder holds only the model's chat format (chat.rs); a new chat format is a new folder here and one line
//! in qwen/serve.rs. inkling2: Inkling 2's looped network, its chat format and its server half.

/// The architectures (GGUF general.architecture) the qwen engine runs
pub const QWEN_ENGINE: [&str; 8] =
    ["qwen35moe", "qwen3next", "qwen35", "qwen4exp", "qwen3moe", "ernie4_5-moe", "deepseek2", "lfm2moe"];

/// Every architecture torc-server runs: the qwen engine's, Gemma 4, gpt-oss and Inkling 2 (`torc-server --archs`
/// prints them)
pub const ARCHS: [&str; 11] = [
    "qwen35moe",
    "qwen3next",
    "qwen35",
    "qwen4exp",
    "qwen3moe",
    "ernie4_5-moe",
    "deepseek2",
    "lfm2moe",
    "gemma4",
    "gpt-oss",
    "inkling2_loop",
];

pub mod ernie;
pub mod gemma;
pub mod glm;
pub mod gptoss;
pub mod inkling2;
pub mod lfm2;
pub mod moonlight;
pub mod qwen;

#[cfg(test)]
mod tests {
    #[test]
    fn every_architecture_is_listed() {
        assert!(super::QWEN_ENGINE.iter().all(|a| super::ARCHS.contains(a)));
        assert!(super::ARCHS.contains(&"gemma4") && super::ARCHS.contains(&"gpt-oss"));
    }
}
