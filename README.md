<p align="center"><img src="docs/images/torc-logo.jpg" alt="ToRc. Big models, small machines." width="640"></p>

# ToRc

[![CI](https://github.com/Rendeverance/ToRc/actions/workflows/ci.yml/badge.svg)](https://github.com/Rendeverance/ToRc/actions/workflows/ci.yml) [![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](#licence) [![Release](https://img.shields.io/github/v/release/Rendeverance/ToRc)](https://github.com/Rendeverance/ToRc/releases/latest) [![Works with ToolFunnel](https://img.shields.io/badge/works%20with-ToolFunnel-8250df)](https://github.com/Rendeverance/toolfunnel)

*By Rendeverance.*

ToRc started by accident. I had two base Snapdragon X mini PCs left over after some testing on Windows on ARM, and I
wanted to see if I could get a big model running well on them. What I got from the existing tools surprised me enough
that I wanted to see how far it would go, and it quickly became a passion project. In the end I thought it would be
useful to others. I hope it helps you. If you want to help ToRc grow, please look at [CONTRIBUTING.md](CONTRIBUTING.md).

<p align="center"><img src="docs/images/torc-demo.gif" alt="ToRc in a few seconds: download the zip, pick a model, talk to it" width="800"></p>

**What is ToRc?** ToRc is a fast CPU inference engine built in Rust, primarily for mixture-of-experts language models.
It will work with small dense models, but it shines with mixture-of-experts models. It uses its own model format. I
built it to run big models on machines that don't have a big GPU, and to be able to trust what it says: on the CPU
you get the same logits at any thread count and any prompt block size, so every speed-up is checked bit for bit
against the version before it.

It runs on Windows, Linux and macOS, on x86-64 (AVX2, AVX-VNNI or AVX512-VNNI) and ARM64 (NEON, dotprod, i8mm). If you have
a Snapdragon NPU it reads prompts with it, a Vulkan GPU takes a share of a long prompt's attention, and a second
machine can join in as a pipeline worker.

**ToRc is more than just an engine.** The zip contains a small app that lets you easily and quickly convert a model,
tune it to your machine and start chatting right away, with saved chats, primers, tool calling and MCP. You can add
your own MCP servers, or put them behind [ToolFunnel](https://github.com/Rendeverance/toolfunnel), my **zero-dependency**
MCP and local tool gateway. ToolFunnel is great for small models because it provides a lean register of tools and MCP servers and
gates every call through your own policy hooks, so you get not only tool use but hooks with your models, without
drowning a local model's small context.

> [!IMPORTANT]
> **Linux and macOS support is currently experimental.** Windows has had extensive testing, on x86-64 and on
> Snapdragon X, and every speed in [PERFORMANCE.md](PERFORMANCE.md) was measured there. Linux and macOS build and run,
> but they have had far less testing due to the equipment that I have access to, so you may see lower speeds or rough
> edges (hopefully not). If you run ToRc on either, I'd really like to hear how it goes, and contributions would be
> brilliant ([CONTRIBUTING.md](CONTRIBUTING.md)).

## Get started in a few minutes

The whole of ToRc, the engine and the desktop app, is a zip of under 20 MB with nothing to install on Windows (its
window is Microsoft's WebView2, which Windows 11 and an up-to-date Windows 10 already have). On Linux the app uses the
system's WebKitGTK, which most desktops already have (if yours hasn't, on Ubuntu 22.04 or later and Debian 12 or later:
`sudo apt install libwebkit2gtk-4.1-0`). If you can unzip a file, you can run your own model on your own machine:

1. **Download the zip for your machine from the [Releases page](https://github.com/Rendeverance/ToRc/releases/latest)**
   and unzip it somewhere of your own, such as Documents. On a Mac the zip isn't signed, so macOS will block it: after
   unzipping, run `xattr -dr com.apple.quarantine <the ToRc folder>` in Terminal.
2. Open ToRc.
3. Pick a model. A GGUF is converted once, in the app, to ToRc's format. GGUFs are on Hugging Face: search for the
   model's name with "GGUF". Most of the speeds here are Q4_0, so that's a good place to start.
4. Press Auto-tune once: it measures your machine and keeps the fastest settings.
5. Load it and talk.

The folder holds everything the app keeps (settings, chats, primers, the MCP servers you add), so moving ToRc is
moving one folder. What the engine learns about your machine (Auto-tune's results, the NPU's busy experts, the
two-machine split) is kept apart, in `%LOCALAPPDATA%\torc` on Windows and `~/.cache/torc` on Linux and macOS, so a new
version picks it up. To remove every trace of ToRc, delete that folder too. [GUIDE.md](GUIDE.md) covers building it yourself, serving the engine to other clients and
tuning it by hand; [AGENTS.md](AGENTS.md) is the same for an AI agent helping you set it up. Every engine setting is a plain
switch listed in [SWITCHES.md](SWITCHES.md), so an agent driving `torc-server` and `torc-bench` can also search for better settings
on your hardware.

### Models

`torc-server` serves all of these, with saved chats and tool calls.

| Model | Architecture | Tool calls written in |
|---|---|---|
| Qwen3.6-35B-A3B | `qwen35moe` | Qwen3.5's format |
| Qwen3-30B-A3B | `qwen3moe` | Qwen3.5's format |
| Qwen3.8 Flash Next (larger than RAM, two boxes) | `qwen4exp` | Qwen3.5's format |
| Qwen3-Next 80B, Qwen3.5 122B | `qwen3next` | Qwen3.5's format |
| The dense Qwen3.5 models | `qwen35` | Qwen3.5's format |
| ERNIE-4.5-21B-A3B | `ernie4_5-moe` | Qwen2.5's JSON format (its template has none) |
| GLM-4.7-Flash | `deepseek2` | its own template's format |
| Moonlight | `deepseek2` | Qwen2.5's JSON format (its template has none) |
| LFM2.5-8B | `lfm2moe` | its own template's format |
| Gemma 4 26B-A4B | `gemma4` | its own template's format |
| gpt-oss | `gpt-oss` | its own template's format |
| Inkling 2 Mini (text) | `inkling2_loop` | its own template's format |

Quant types: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL, IQ4_XS, IQ3_S, Q3_K, Q4_K, Q5_K, Q6_K, MXFP4, and Q2_K, Q2_0,
IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS and IQ1_M through portable kernels that run on any CPU (Q2_K and Q2_0 with vector code
on x86 and ARM; the others are slower than the types with kernels of their own).

Any GGUF of a supported architecture should work, including fine-tunes and abliterated versions, as long as its quant
types are in the list. For Inkling 2, use a mainline conversion (the `inkling2_loop` architecture, as in
[ggml-org's IQ4_NL and Q8_0 files](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF)); the `inkling2` files from
Unsloth's own runtime, and its NVFP4 expert tensors, are not supported yet. A Mainline_Support file's `.gguf_file` name
is accepted as a GGUF too. The app checks before converting that the model will run and tells you why if it won't;
`torc plan <model.gguf>` does the same from the command line.

For replies, active parameters matter, not the file size: the 35B-A3B writes faster than Gemma's 26B-A4B on the same
machine, because each token reads about 3 billion weights instead of 4.

## The engine without the app

The app is optional. The zip's `engine` folder holds `torc-server`, which runs on its own and speaks the OpenAI and
Anthropic APIs, so any chat app, agent framework or script can use it:

```
engine/torc-server models/your-model.torc 8080 auto 16384
```

That's the model, the port, the threads and the context. Then set your client up with:

- base URL `http://127.0.0.1:8080/v1` (without the `/v1` works too)
- model: the name `GET /v1/models` lists, which is the file's name without `.torc`
- API key: anything, or none, on the same machine

To use it from other machines, start it with two switches set:

```
TORC_HOST=0.0.0.0 TORC_API_KEY=a-long-random-key engine/torc-server models/your-model.torc 8080 auto 16384
```

(in PowerShell, `$env:TORC_HOST = '0.0.0.0'; $env:TORC_API_KEY = 'a-long-random-key'` first). Other machines then
use `http://<this machine's address>:8080/v1` with that key as their API key, while this machine still needs none.
The traffic isn't encrypted, so keep it to a network you trust. Run Auto-tune once first (in the app, or
`engine/torc-tune models/your-model.torc`) so the server starts with this machine's fastest settings.
[GUIDE.md](GUIDE.md) has the rest: warming the model up, drafting, sampling and every switch.

## How fast

Qwen3.6-35B-A3B Q4_0, **<ins>a 4k-token prompt</ins>** and a 128-token reply, in tokens a second as the server times them.

| Machine | Read | Write |
|---|---|---|
| Two Snapdragon X mini PCs (32 GB each) over USB4, CPU + NPU + GPU | 406 | 54 |
| One of them, CPU + NPU + GPU | 283 | 55.7 |
| Desktop, i5-12600K, CPU only | 182 | 41 |
| The same desktop, llama.cpp b12360 at its best thread count | 107 | 21 |

The top row is two machines working together. On one machine, expect the second row, a little lower if its clocks drop
under a long load (thermals: we haven't seen that on the IdeaCentre mini PCs, but it can happen on a laptop).

**Bigger than memory: Qwen3.8 Flash Next, a 107 GB file, on the same two boxes with 64 GB of RAM between them, the
experts that don't fit read from the SSDs as they are needed. About 67 tokens a second reading a 16k-token prompt and
17 writing.**

**Every model and machine measured so far, with llama.cpp beside it where it was run: [PERFORMANCE.md](PERFORMANCE.md).**

<p align="center"><img src="docs/images/two-boxes.jpg" alt="The two Snapdragon X mini PCs, joined by a USB4 cable" width="560"></p>

ToRc's write figures are with its drafts on, which is how it runs by default. llama.cpp is shown without drafts:
with `--spec-type draft-mtp` it measured between slower and level on this machine.

[PERFORMANCE.md](PERFORMANCE.md) has every prompt size for each model and machine, and [RESULTS.md](RESULTS.md) says how
they are measured and how to measure your own machine. Longer prompts read slower. With drafts on, the write speed
moves by a few tokens a second from one reply to the next, and more on long prompts.

## What's in it

- **Kernels** (`kernels/`): every quant type's products for one token and for many, NEON SDOT/SMMLA on ARM and
  AVX2/VNNI on x86. The parity tests hold every path of a kernel to the same bits, and each to the scalar
  definition of its type within rounding.
- **Attention** (`model/src/layers/fattn.rs`): built from its definition in [SPEC.md](SPEC.md), the same bits alone or in any
  block.
- **Prompt path**: a block's products read each weight once for all its tokens; experts grouped by token; reused
  buffers; a phase profiler (`TORC_PROF=1`).
- **Expert cache** (`model/src/runtime/ecache_own.rs`): the routed experts of a model bigger than RAM, read from the SSD
  with overlapped I/O, warmed from a per-machine usage profile.
- **Two boxes** (`model/src/runtime/pipe.rs`): a layer pipeline over USB4, the worker keeping its layers' state, prompts
  overlapped block by block, each machine loading only its own layers.
- **NPU tier** (`model/src/tiers/npu.rs`, `npu_gemma.rs`): dense projections, the shared expert and the hottest experts
  of prompt blocks on the Snapdragon NPU, overlapped with the CPU; Gemma 4 whole on the NPU. It needs a ggml build with
  the Hexagon backend, which the GenieX CLI installs (see [GUIDE.md](GUIDE.md)).
- **GPU tier** (`model/src/tiers/gpu.rs`, the `gpu` feature, `TORC_GPU_ATTN=1`): a long prompt block's attention shared
  between a Vulkan GPU (integrated or discrete, through wgpu) and the CPU at the same time, the split following the
  two sides' measured speeds and kept for the next run. A GPU that turns out to be no gain over the CPU alone is
  switched off for that model.
- **MTP**: the model's own prediction block as drafter, revertible DeltaNet state, and a draft planner that picks
  how many tokens to draft on each pass from what a pass costs on this machine. Every draft is verified by the full
  model before it is kept, so drafts never change the output (at the default settings the NPU and GPU never take
  those passes).
- **Fast mode** (`TORC_TAIL_SKIP`, off by default): skips the lowest-ranked experts that are not in RAM. Not
  exact: on 1,140 MMLU questions (20 a subject, thinking off, greedy) with Qwen3.8 Flash Next it scored 86.1%, the
  same as exact, and gave the same answer to 98% of them.
- **Server** (`torc-server`): OpenAI chat completions and responses, Anthropic messages, prompts never read twice
  (states kept in memory and on disk).
- **Format** ([`FORMAT.md`](FORMAT.md)): each expert one aligned read, every byte traceable to its GGUF, BLAKE3 + CRC-32C.
- **hwscan**: hardware inventory and speed tests.
- **torc-bench** (`tools/bench/torc-bench.js`): the standard prompts against a running server, the results as a CSV
  ([RESULTS.md](RESULTS.md)).
- **ToRc** (`desktop/`, its page in `ui/renderer/`): the desktop app, a small Rust shell (Tauri) around the system's
  own web view. It converts GGUFs, tunes, starts torc-server for you, and has a chat with saved chats, primers,
  compacting, built-in tools (the time, a calculator, the model's own context use, compacting and wake-ups),
  your own MCP servers or ToolFunnel, and every switch one click away.

## Exactness

Integer sums are exact whatever the order, so the kernels reorganise them freely (rows in vector lanes, 2x2 SMMLA
tiles, state rows interleaved). Every floating-point step per value (scales, fused multiply-adds, the final lane
sums, fp16 rounding) keeps a fixed order. That makes a prompt read in blocks give the same logits as token by
token, and every change is checked by comparing whole logit dumps (`torc-run dump=`, then `cmp`).

The NPU tier is not exact by design: the NPU works on unrounded f32 activations. It is checked by tolerance
against the CPU path (`torc-run ref=`). Decode and MTP verification never use it, and with it off the output is
byte-identical.

The GPU tier is not bit-exact either: its sums run in another order than the CPU's, so its attention agrees with
the CPU's to rounding (about 1e-5 relative at 32k tokens). It only runs on prompt blocks, `TORC_GPU_CHECK=1`
compares every block it runs against the CPU, and with it off the output is byte-identical. How a block is split
between the GPU and the CPU follows their measured speeds, so with the GPU tier on, two runs of the same prompt can
give different replies where the model's choice is a near tie.

Sampling stays exact too: a sampled reply (temperature, top_p, top_k, presence and frequency penalties) with a fixed
`seed` is the same tokens with drafting on or off, because every drafted token is checked against exactly the pick
normal decoding would make at that point, penalties included.

## Building

```
# ARM64 (Windows on ARM: the default MSVC toolchain, or aarch64-pc-windows-gnullvm with llvm-mingw)
cargo build --release --workspace
# x86-64 (Windows, rustup's gnu toolchain; needs AVX2 and FMA)
./build-x86.sh
# with the GPU tier (needs a Vulkan driver at run time)
cargo build --release --workspace --features torc-model/gpu      # ARM
./build-x86.sh target --features torc-model/gpu                  # x86-64
```

The llvm-mingw build imports `libunwind.dll`: ship it beside the executables.

## Running

The programs you use:

- `torc-server <model> [port] [threads] [ctx] [name]`: the chat server
- `torc-tune <model> [threads] [ctx]`: measures this machine once and keeps the fastest settings
- `torc-chat <model> "message"`: one reply in the terminal, for a quick check of a Qwen model
- `torc-run`: the engine with no server in front, and the worker on the second machine of two (`serve-pipe=<port>`). It
  reads a file of token ids and writes each position's logits, which is how every faster path is checked against the
  plain one:
  `torc-run <model> <tokens.int32> [n] [threads|auto] [block=N] [nologits] [dump=logits.f32] [ref=reference.f32]`
- `torc convert-gguf <in.gguf> <out.torc>`: makes a ToRc file (`torc` also checks and describes one, and
  `torc plan <model>` says whether a model will run, and if not, why)
- `hwscan`: what this machine has (CPU, memory, GPU, NPU), for the benchmark's CSV. The full inventory is Windows-only
  for now: on a Mac it knows the CPU's name, the memory and the CPU count, on Linux the CPU count and little else

The developer tools (the NPU and GPU checks, drive probes, the regression check across models, a tokenizer dump and
others) are in `model/src/bin/dev`; each says what it does in its first lines.

`auto` threads = every logical CPU of the fastest core class (the P-cores of a hybrid CPU), every core on ARM. On Linux
`auto` can't tell the core classes apart yet and takes every CPU, so on a hybrid CPU (Intel 12th gen on, Ryzen AI) the
E-cores set the pace: give the P-cores' thread count by hand. Environment: `TORC_WARM=wait` loads the model into RAM
before serving; `TORC_PROF=1` prints phase times; `TORC_NPU=1` turns the NPU tier on (ARM64 Windows; `npucheck` is its
smoke test); `TORC_PIPE=host:port` sends the later layers to a worker started with
`torc-run <model> ... serve-pipe=<port>`.

### Prompts not read twice (torc-server)

torc-server reads only what is new in a prompt. In order, a request starts from:

1. the state of the last request, when its prompt extends it;
2. the last prompt's mark (where its messages ended): the next turn of the same chat;
3. the longest state saved on disk the prompt begins with, when it is more than 512 tokens past (4): a chat
   coming back;
4. the system prompt's mark, when the cache still holds that system prompt: a new chat with the same primer;
5. the longest state saved on disk the prompt begins with (a chat's earlier turns, or its system prompt);
6. zero (and the system prompt's state is saved to disk once read).

States go to disk (`markfile.rs`) when a request switches to another chat (the one left, a conversation of two
turns or more, from 512 tokens), after `TORC_IDLE_SAVE` seconds without a reply (default 60, 0 = never), and when
a system prompt of 64 tokens or more is read from zero. Each box keeps its own layers' half in `TORC_MARK_DIR`
(default `marks` beside the executable), at most `TORC_MARKS` files (default 16, the oldest removed; 0 = off). A
file is checked against the model's identity and a BLAKE3 hash before it is used, and on the CPU the replies are the
same tokens as a prompt read from zero. The NPU and GPU tiers are not bit-exact, so with one of them on a chat
picked up from a saved state can differ where the model's choice is a near tie.

`usage.prompt_tokens_details.cached_tokens` says how much of a prompt was not read.

Gemma 4 and gpt-oss go through the same server and the same three APIs, with tool calls in each model's own format
and chats saved by the same rules. Their whole state is the attention cache, so a saved chat can be picked up at any
point and there is no system-prompt mark. They sample as every other model does, and Gemma 4 always answers without
thinking.
Gemma doesn't save chats on the whole-NPU tier (`TORC_NPU=pure`), where the cache lives on the NPU. Gemma 4's output
head is requantised to Q4_0 at load, fewer bytes a token than the file's Q6_K, so its logits are not the file's:
`TORC_HEAD=file` keeps the file's head, `q8_0` takes the middle way.

## Roadmap

- Faster writing at long context: the guessed tokens checked against the attention cache together, so each extra
  guess costs as little at 16k as it does at 1k.
- ~~Linux on bare metal.~~ Done: a user runs it on a Snapdragon X mini PC under Ubuntu, on bare metal, and it
  builds and passes its tests under WSL2 (Ubuntu 24.04) too. The Snapdragon NPU works there as well, with llama.cpp's
  Hexagon backend installed.
- macOS on Apple Silicon: EXPERIMENTAL. One person has built it from source and run it. The release may have a
  macos-arm64 zip built by GitHub, but I don't have a Mac to test it myself. The zip isn't signed, so macOS will block
  it: after unzipping, run `xattr -dr com.apple.quarantine <the ToRc folder>` in Terminal. If you have a Mac, I'm
  eager to hear from you: please test it and get in touch.
- Experts on the GPU. Today the GPU only takes a share of a long prompt's attention.
- NPU and GPU together carrying the prompt with the CPU nearly free.
- Running on the NPU and GPU alone (integrated or discrete), leaving the CPU to you, so the machine stays usable
  while the model works. This needs the experts on the GPU first.
- One copy of the NPU's hot experts instead of two (about 1 GB back on a 32 GB machine).
- Intel and AMD NPUs. Current support includes Snapdragon X NPUs. Intel and AMD NPU support is planned when I have
  access to suitable hardware or contributors can assist.
- A fast attention option for long prompts, offered only once its agreement with the exact path is measured at longer
  contexts and the quality is acceptable.
- Several replies written at once. The server writes one at a time today, and other requests wait their turn.
- File tools: the model writing files into the workspace folder, asking you before each write, built as ToolFunnel
  tools.
- Vision: models that read images, and pictures in the desktop app's chats.
- Speech and audio: talking to a model and hearing it reply, and models that take audio in.

## Licence

ToRc is dual-licensed under the MIT licence (LICENSE-MIT) or the Apache License 2.0 (LICENSE-APACHE), at your option.
Copies keep the copyright line, and under Apache the NOTICE file with it. [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) lists the code, data
and ideas ToRc takes from other projects (ggml and llama.cpp, Strata, Cephes, papers), and [THIRD-PARTY-LICENSES.md](THIRD-PARTY-LICENSES.md) every
crate a build compiles, with its licence text (all of them permissive: MIT, Apache-2.0, Zlib, ISC, BSD-2-Clause,
Unicode-3.0). Models keep their own licences; this repository ships no weights.
