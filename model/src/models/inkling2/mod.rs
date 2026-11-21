//! inkling2 - Inkling 2 (Mainline_Support GGUFs, arch `inkling2_loop`): the text model (model.rs), its
//! hyperparameters (hparams.rs), its chat format (chat.rs) and its server half (serve.rs).

pub mod chat;
pub mod hparams;
#[allow(clippy::module_inception)]
pub mod model;
pub mod serve;
