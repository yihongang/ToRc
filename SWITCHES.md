# Environment switches

ToRc runs well with none of these set: every default is chosen for the machine it finds. An environment variable
always wins over a tuned value ([AUTOTUNE.md](AUTOTUNE.md)). The bits a model produces never depend on the performance settings
below; the ones that do change output say so.

Most of what is listed here you will never need. Each switch is marked with its kind:

- **setting**: something you might set for your machine or your files. The guides cover the ones that matter.
- **diagnostic**: prints or writes something to look at.
- **development**: a code path, the off switch for one, or a threshold, left in from development. You can run ToRc
  without any of them. I kept them because each is working code that may earn its place on hardware I haven't tried,
  and switching a path off is how a default gets checked against what it replaced.

The switches are read once, when the program starts, so changing one means starting it again.

## Threads and the pool

| Switch | Kind | What it does |
|---|---|---|
| `TORC_POOL` | development | worker pool implementation |
| `TORC_QOS` | setting | macOS only: `0` leaves the engine's threads in the class macOS gives them; by default they are marked as work someone is waiting for (user-initiated), so macOS doesn't schedule them as background work |
| `TORC_PIN` | setting | `0` leaves worker threads unpinned, `1` pins them; pinned by default on Windows only (Linux and macOS leave it to the OS); `1` pins on Linux too (experimental there, though a user has measured it faster on a Snapdragon X under Ubuntu), and changes nothing on macOS, which has no way to pin a thread. Unset, on Windows, Auto-tune's measured choice for the model is used when there is one (AUTOTUNE.md) |
| `TORC_DECODE_THREADS` | setting | threads a reply step uses (the decode width; default: the tuned `decode_threads`, else all). On a part with two threads a core, one a core can write faster while the prompt still reads on all |
| `TORC_SPREAD` | development | `0` uses CPUs 0..n instead of spreading over the fastest cores |
| `TORC_SPIN_US` | development | microseconds a worker spins before it parks (default 200) |
| `TORC_TIMER_1MS` | development | asks Windows for a 1 ms timer so short waits are not rounded up to ~15.6 ms |
| `TORC_GUIDED` | development | `0` turns guided scheduling of work items off |
| `TORC_FLOW` | development | `0`: a fork-join per stage instead of flowing items between stages |
| `TORC_BLOCKED` | development | `0`/`1` overrides blocked item order (on by default on ARM64) |
| `TORC_DYN`, `TORC_MOE_DYN` | development | `0`/`1` overrides the dynamic work split (used where cores differ in speed) |

## Prompt blocks and batching

| Switch | Kind | What it does |
|---|---|---|
| `TORC_BLOCK` | setting | the prompt block size the server reads with (else the tuned value, else the rule: 512 on the CPU, 256 with the NPU, or 1024 with the NPU on the Qwen3.6-35B and the other Qwen3.5 / 3.6 MoE models when the whole model is in memory, and 128 on two machines) |
| `TORC_BLOCK_LONG` | setting | block size for long prompts |
| `TORC_CHECKPOINT`, `TORC_CHECKPOINT_MIN` | development: `TORC_CHECKPOINT`; the rest setting | state checkpoints while reading a prompt (retry granularity; the fewest tokens worth one) |
| `TORC_GEMM_MIN`, `TORC_PACKED_MIN` | development | from how many vectors the packed matrix kernels take over |
| `TORC_MULTI`, `TORC_GROUPED`, `TORC_GROUPED_MAX` | development | multi-vector and grouped expert products |
| `TORC_RC`, `TORC_RC_WIDE` | development | rows per task in the row-split products |
| `TORC_FUSE` | development | fuses the down products into the gate/up sweep |
| `TORC_DENSE_DYN` | development | dynamic split of dense products |

## Kernels (per CPU)

| Switch | Kind | What it does |
|---|---|---|
| `TORC_DOT`, `TORC_DOTN` | development | dot-product kernel variant for router and dense rows |
| `TORC_KMM`, `TORC_KMM_MIN`, `TORC_KMM8_MIN`, `TORC_KMM_PANEL`, `TORC_KMM_X86_MIN` | development | K-quant matrix kernels and their thresholds |
| `TORC_MMLA`, `TORC_MMLA_MIN`, `TORC_MMLA8_MIN`, `TORC_MMLA_PANEL`, `TORC_VPANEL` | development | ARM i8mm (SMMLA) kernels and their thresholds |
| `TORC_Q4X4`, `TORC_Q4X4_EXPERTS`, `TORC_Q4X4_HEAD`, `TORC_Q4X4_MIN`, `TORC_Q4X4_MAX` | setting: `TORC_Q4X4`; the rest development | repacked Q4_0 layout for i8mm machines |
| `TORC_Q6_MIN` | development | from how many vectors Q6_K uses the matrix kernel |
| `TORC_VNNI`, `TORC_X86_WIDE`, `TORC_X86_PF_BLOCKS` | setting: `TORC_X86_PF_BLOCKS`; the rest development | x86 kernel choices (`TORC_VNNI=0`: no VNNI, in either its AVX or its AVX-512 form); software prefetch distance in blocks (default 16, `0` off) |
| `TORC_GDN`, `TORC_REC_ROWS` | development | `0` uses the portable DeltaNet step; state rows per recurrence item |
| `TORC_MOE_DRC`, `TORC_MOE_GRC`, `TORC_MOE_JG` | development | row chunking of the expert products |
| `TORC_HC_TILE`, `TORC_HC_FUSEQ`, `TORC_MIX`, `TORC_MIX_KB`, `TORC_MIX_SPLIT`, `TORC_MIX_MM` | setting: `TORC_MIX_MM`; the rest development | hyper-connection (Flash Next) mixer kernels and tiling; `TORC_MIX_MM=1`: a prompt's mixer products with SMMLA (ARM with i8mm) |
| `TORC_QSA_GATHER` | development | `0` turns the decode gather of the attention off |
| `TORC_ATTN_FAST` | setting | `1`: attention scores over each query rounded to sixteen bits, an exact integer sum against the keys: faster at long context, and no longer bit-identical to the default (the scores differ by about 3e-5 of the largest). Off by default. For models with an 8-bit cache and heads of up to 256 |
| `TORC_ATTN_F32`, `TORC_ATTN_F32_MIN`, `TORC_ATTN_TILE`, `TORC_ATTN_TILE_MIN`, `TORC_ATTN_ROWS2` | setting: `TORC_ATTN_TILE`; the rest development | attention page conversion to f32 and tiling (the tiled path: ARM only, on by default for the Qwen3.6-35B's architecture, `TORC_ATTN_TILE=0` or `1` to choose) |
| `TORC_GGML_REF` | development | the kernel tests ask ggml itself as well: path to the harness `scripts/ggml_ref.sh` builds from a ggml checkout. Unset, those tests say so out loud and pass, so a machine without ggml still builds green |

## Memory and disk (models bigger than RAM)

| Switch | Kind | What it does |
|---|---|---|
| `TORC_CACHE_MB` | setting | expert cache budget (`0` off; a size streams the experts through a cache of that size, even for a file that fits; by default on when the file does not fit in memory) |
| `TORC_CACHE_LAYERS` | setting | `a-b`: the layers this machine loads ahead (expert cache warm set, NPU hot experts); on two machines each one's own half by default ([TWO-BOX.md](TWO-BOX.md)) |
| `TORC_IO`, `TORC_IO_THREADS`, `TORC_IO_DEPTH`, `TORC_IO_PIECE_KB`, `TORC_IO_SORT`, `TORC_IO_BUFFERED` | development: `TORC_IO`, `TORC_IO_BUFFERED`; the rest setting | how expert reads are issued ([EXPERT-CACHE.md](EXPERT-CACHE.md)) |
| `TORC_OWN`, `TORC_OWN_LARGE`, `TORC_OWN_PREFETCH` | development: `TORC_OWN`; the rest setting | an owned copy of the weights that are always in memory; that copy on large pages (it turns the copy on too); `0` turns the expert cache's read-ahead off |
| `TORC_LARGE_PAGES`, `TORC_EXPERTS_LARGE`, `TORC_KLAY`, `TORC_KLAY_HEAD`, `TORC_KLAY_LARGE`, `TORC_KLAY_LAZY`, `TORC_KLAY_Q6` | setting: `TORC_LARGE_PAGES`, `TORC_EXPERTS_LARGE`, `TORC_KLAY_LARGE`; the rest development | large pages for the expert cache (on by default on Windows, `0` turns them off), for the experts of a model that fits, and for the laid-out copies of K-quant matrices (ARM) |
| `TORC_WARM`, `TORC_WARM_PROFILE` | setting: `TORC_WARM`; the rest development | reading the file into memory at load: `0` skips it, `1` forces it, `wait` does it before requests are taken (default: when the model fits); `TORC_WARM_PROFILE=0` skips warming the expert cache from the usage file |
| `TORC_READ_AHEAD`, `TORC_READ_AHEAD_BLOCK` | development | read-ahead of the next layer's likely experts (off by default) |
| `TORC_TAIL_SKIP` | setting | FAST MODE, changes output: `r` skips an expert that every token ranks r or lower (from 1) and that is not in RAM (off by default) |

## Two machines

| Switch | Kind | What it does |
|---|---|---|
| `TORC_PIPE`, `TORC_PIPE_SPLIT`, `TORC_PIPE_PULL`, `TORC_PIPE_TIMEOUT`, `TORC_PIPE_FALLBACK`, `TORC_PIPE_TOPK` | development: `TORC_PIPE_TOPK`; the rest setting | the layer pipeline over two machines ([TWO-BOX.md](TWO-BOX.md)) |
| `TORC_PIPE_BIND` | setting | on a worker: the address it listens on (default: every interface; a worker has no authentication) |
| `TORC_PIPESPEC` | development | `0` turns pipelined speculation off |
| `TORC_EP`, `TORC_EP_MIN_N` | development | expert-parallel worker and the smallest block worth sending |
| `TORC_HEAD_SPLIT` | development | splits the output head across the machines |

## NPU (Snapdragon Hexagon, through a ggml build)

| Switch | Kind | What it does |
|---|---|---|
| `TORC_NPU`, `TORC_NPU_DIR`, `TORC_GEMMA_NPU` | setting | turns the NPU tier on (changes output: not bit-exact, prompts only; Inkling 2's hot routed experts can go up too but measured 0.27x the CPU at one token, so leave them off; `pure` runs Gemma 4 whole on the NPU); where the ggml build is; Gemma's NPU parts |
| `TORC_NPU_MIN_N`, `TORC_NPU_BLOCK`, `TORC_NPU_GROUP` | development | smallest block for the NPU (Inkling 2 needs `1`, its blocks are one token, and they measured slower there); longest block per graph; layers per buffer (Gemma 4 whole on the NPU) |
| `TORC_NPU_DEV`, `TORC_NPU_BUF_MB` | development | the ggml device to open, by name (default `HTP0`); the largest NPU weight buffer in MiB (default 256 on Linux, `0` one buffer, the Windows default) |
| `TORC_NPU_MAX_MB` | development | the most weight bytes to put on the NPU, in MiB (default `0`, no limit): what is past it stays on the CPU. For a Linux driver that maps the NPU's buffers inside a small address window |
| `TORC_NPU_PROFILE`, `TORC_NPU_HOT`, `TORC_NPU_HOT_MIN`, `TORC_NPU_PAD` | setting: `TORC_NPU_PROFILE`, `TORC_NPU_HOT`; the rest development | hot experts on the NPU: a usage file to take them from (default on one machine: the one this machine keeps for the model), how many a layer (16, or 32 on one machine with the Qwen3.6-35B and the other Qwen3.5 / 3.6 MoE models when the whole model fits with 8 GB to spare; `0` none) |
| `TORC_NPU_ASYNC`, `TORC_NPU_GUIDED`, `TORC_NPU_WARM`, `TORC_NPU_Q8`, `TORC_NPU_HEAD`, `TORC_NPU_HELPER_PRIO` | development | scheduling and form of the NPU work |
| `TORC_NPU_SHARED` | development | `0` keeps the shared expert on the CPU |
| `TORC_INKLING_MOE_PAR` | development | `0` runs Inkling 2's routed experts one at a time on the calling thread instead of one thread an expert (the bits are the same either way) |

## GPU (Vulkan, through wgpu; in the release zips, or built with the `gpu` feature)

| Switch | Kind | What it does |
|---|---|---|
| `TORC_GPU_ATTN` | setting | `1`: long prompt blocks' attention shared between the GPU and the CPU (not bit-exact, prompts only) |
| `TORC_GPU` | setting | the adapter, by a piece of its name (default: the first integrated GPU, else the first GPU) |
| `TORC_GPU_SHARE` | setting | the GPU's share of each block's work, 0..1 (default: learned per machine, GPU, kernel and attention shape, kept in the tune file) |
| `TORC_GPU_KERNEL` | development | `generic`: every shape on the generic kernel, to measure a shape's own kernel against it (default: the kernel `gpu.rs` lists for the shape) |
| `TORC_GPU_SLICE` | development | the most work (query heads x keys) in one GPU submission (default: about half a second of this GPU's measured speed) |
| `TORC_GPU_MIN`, `TORC_GPU_FROM` | development | smallest block it takes (32); fewest keys the block's last token must see (4096) |
| `TORC_GPU_SPLIT`, `TORC_GPU_WG` | development | a block's keys split into this many parts that run side by side (default: chosen per block, `1` no split); the workgroups the split aims for (64) |
| `TORC_GPU_CHECK` | setting | `1`: runs the GPU's tokens again on the CPU and prints the largest difference, the kernel and the slices |
| `TORC_GPU_STATS` | diagnostic | `1`: prints where the GPU tier's time went, every 256 blocks |

## Drafting (MTP) and serving

| Switch | Kind | What it does |
|---|---|---|
| `TORC_MTP`, `TORC_MTP_FILE`, `TORC_MTP_HEAD`, `TORC_MTP_FILL_FULL` | setting: `TORC_MTP`, `TORC_MTP_FILE`; the rest development | the most MTP drafts a pass (default 8, at most 16, `0` off); a separate MTP block file; draft head; full-block fill |
| `TORC_DRAFT`, `TORC_DRAFT_N`, `TORC_DRAFT_P`, `TORC_DRAFT_HEAD`, `TORC_DRAFTS` | setting: `TORC_DRAFT`; the rest development | Gemma's drafter (the path to its assistant model); drafts per pass; fixed threshold; its head; a file for the planner's calibration (torc-run only) |
| `TORC_HOST`, `TORC_API_KEY`, `TORC_IDLE_SAVE`, `TORC_IDLE_SAVE_MIN` | setting | server bind address; the key another machine's requests must carry, as `Authorization: Bearer` or `x-api-key` (this machine's never need it; set it whenever `TORC_HOST` opens the server to a network); seconds without a request before the current chat's state is saved (default 60, `0` never); the fewest prompt tokens a one-message chat needs to be saved then (default 8192) |
| `TORC_MARKS`, `TORC_MARK_DIR` | setting | saved conversation states kept per machine (default 16) |
| `TORC_TOKENIZER`, `TORC_VOCAB_MAP`, `TORC_NGRAM_IDMAP` | setting | a vocabulary-sliced model's original tokenizer and id maps |
| `TORC_HEAD` | setting | changes output: Gemma 4's output head at load (default `q4_0`; `q8_0`, or `file` for the file's own head) |
| `TORC_TUNE_LENS`, `TORC_TUNE_TEXT` | setting | torc-tune's prompt lengths and text |

## Diagnostics

| Switch | Kind | What it does |
|---|---|---|
| `TORC_PROF` | diagnostic | `1`: where the time goes, per phase |
| `TORC_TRACE` | diagnostic | `1`: one line per layer about the hidden state (to see where two runs part) |
| `TORC_DUMP` | diagnostic | a folder: each layer's state written there as `<layer>.f32`, to compare with a reference |
| `TORC_TRACE_ROUTES` | diagnostic | `1`: each token's best six experts with their router logits |
| `TORC_ROUTER_F32` | diagnostic | `1`: router logits from an f32 copy of the router, a test of the fp16 copy's rounding (changes the bits) |
| `TORC_BENCH_NT`, `TORC_BENCH_OFF` | diagnostic | the kernel benches in `kernels/tests`: threads, and the first CPU they are pinned from |
| `TORC_METRICS` | diagnostic | server per-request timings |
| `TORC_ROUTES`, `TORC_NPU_PROFILE_OUT` | diagnostic | write every call's expert routes; write a usage file of expert counts |
| `TORC_PLANLOG`, `TORC_DRAFT_LOG`, `TORC_NPU_VERBOSE` | diagnostic | drafting planner and drafter logs; NPU detail |
| `TORC_MTP_FILLHASH` | diagnostic | hash of the MTP fill, for checking it |
| `TORC_ATTN_EXACT` | diagnostic | `1`: prompt attention in f64, the reference the faster paths' rounding is judged against (slow; changes the bits) |
| `TORC_REC_F32`, `TORC_KV16` | diagnostic | `1`: DeltaNet state in f32; attention cache at sixteen bits: tests of the rounding (both change the bits) |
| `TORC_KL_BUCKET`, `TORC_KL_FROM` | diagnostic | torc-kl: mean KL per bucket of positions; score from a position on |
