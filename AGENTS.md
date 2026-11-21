# ToRc for AI agents

This file is for an AI agent helping someone install, configure and run ToRc on their own machine. It covers what to
do and in what order, and what to check before telling the user something works. [GUIDE.md](GUIDE.md) is the same material for
people; [SPEC.md](SPEC.md) describes how each part works inside.

## What ToRc is

A CPU-first inference engine for mixture-of-experts language models, with its own model file format. It serves an
OpenAI-style API on the user's machine. On the CPU the output is deterministic: the same logits at any thread count
and any prompt block size. Windows first (ARM64, and x86-64 with AVX2); Linux runs on bare metal (Ubuntu on a Snapdragon X
mini PC, tested by a user) and under WSL2.

## 1. Check the machine

Run `hwscan` first and read its warnings before anything else. It reports the cores and their classes, memory
bandwidth, real clocks under load, the drives (and which one holds the models), and any NPU. Its inventory is
Windows-only for now: on Linux, read the machine from /proc and /sys instead. Things to tell the user about straight
away:

- the model directory on a USB drive (experts that don't fit in RAM will load at USB speed);
- a clock that drops a lot under sustained load (power mode, thermal limits);
- less RAM than the model file, which means experts stream from the SSD (see section 6).

## 2. Build

    # Windows on ARM: the default MSVC toolchain (VS C++ build tools for ARM64), or aarch64-pc-windows-gnullvm + llvm-mingw
    cargo build --release --workspace

    # Windows x86-64 (AVX2 + FMA)
    ./build-x86.sh

    # either, with the GPU tier: add --features torc-model/gpu (build-x86.sh: after the target dir)

    # Linux: the ARM line, plus a C++ compiler (g++, Ubuntu's build-essential) for the NPU tier's exception guard

The executables land in `target/release`. Built with llvm-mingw, copy its `libunwind.dll` next to them. Run
`cargo test --release` once; the parity tests check every kernel against its definition on this CPU.

## 3. Get a model

ToRc reads its own format, converted from a GGUF (for a split GGUF, name the first part):

    torc plan model.gguf
    torc convert-gguf model.gguf model.torc
    torc check model.torc
    torc check model.torc --against model.gguf
    torc info model.torc

`plan` says whether the model will run here and, if not, why; `convert-gguf` runs it first and refuses a model that
can't. Always run both checks after a conversion or a copy. `check` verifies every hash and checksum; `--against` compares
every tensor and expert with the GGUF byte for byte. A file that fails is not to be used. Models keep their own
licences; ToRc ships no weights.

Supported: Qwen3.6-35B-A3B, Qwen3-30B-A3B, Qwen3.8 Flash Next, Qwen3-Next 80B, Qwen3.5 122B, the dense Qwen3.5 models,
ERNIE-4.5-21B-A3B, GLM-4.7-Flash, Moonlight, LFM2.5-8B, Gemma 4 26B-A4B, gpt-oss and Inkling 2 Mini (the
`inkling2_loop` mainline files; Unsloth's `inkling2` layout and NVFP4 experts are not supported yet), all served by
`torc-server` (it reads the model's architecture from the file). Quant types: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL, IQ4_XS, IQ3_S, Q3_K,
Q4_K, Q5_K, Q6_K, MXFP4, and on any CPU through portable kernels: Q2_K, Q2_0, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ1_M (Q2_K
and Q2_0 with vector code on x86 and ARM).

## 4. Serve it

    torc-server model.torc 8080 auto 16384

The model, the port, the thread count (`auto` = every core of the fastest class) and the context length. Clients
point at `http://127.0.0.1:8080/v1` (`GET /health` for a liveness check, no key needed; `GET /slots` and `GET /metrics` in llama-server's formats); it speaks OpenAI chat completions, the OpenAI Responses API and Anthropic
messages. It listens on this machine only unless `TORC_HOST` is set. Opening it to a network, set `TORC_API_KEY` too:
another machine's requests must then carry it (`Authorization: Bearer <key>` or `x-api-key: <key>`), this machine's
never do. The traffic is plain HTTP, so warn the user before exposing it beyond a network they trust. In the desktop app that is Settings > Share on my network (the app sets
`TORC_HOST` itself and ignores it in the switch list). `TORC_WARM=wait` loads the model before the port opens. For a quick check of a Qwen
model without a client: `torc-chat model.torc "Hello"` (ChatML only; for other families, start the server). A web page
from another site is refused whatever `TORC_HOST` says; non-browser clients send no Origin and are not affected.

The server writes one reply at a time; other chat requests queue in order, while /health, /slots, /metrics and the
model list answer at once. Greedy decoding by default; `temperature`, `top_k`, `top_p`, `seed`,
`presence_penalty` and `frequency_penalty` are honoured. Gemma 4 and gpt-oss are served the same way, with tool
calls and saved chats (not on Gemma's whole-NPU tier). Every model takes tools: GLM, LFM2, Gemma 4 and gpt-oss in
their own templates' call formats, the Qwen models in Qwen3.5's, ERNIE and Moonlight (whose templates have none) in
Qwen2.5's JSON format. The server hands back OpenAI-style `tool_calls` whatever the format.

Control strings in user and tool messages (`<|im_end|>`, `<|im_start|>system`, `<|turn>`) are broken with a word
joiner before the prompt is built (Tokenizer::defuse), so a tool result or a fetched page cannot forge a turn (ERNIE
is the exception: its turns are the plain text "User: " and "Assistant: "). Plain instructions inside that text still
reach the model and can still steer it: tell the user before wiring up tools that can do damage alongside text from
sources they do not control. The system message is read as written.

## 5. Make it fast on this machine

Change one thing at a time and measure each change (section 7). In this order:

1. `torc-tune model.torc` picks the prompt block size for this machine, and whether to pin the threads, and saves both
   where the server finds them. Run it with the same settings the server will use (NPU and GPU on, second machine up).
2. Threads: `auto` is usually right. On a hybrid Intel CPU also try the P-cores only, one thread per core. Worker
   threads are pinned to cores on Windows and left to the OS on Linux and macOS, where the engine can't yet see
   which cores are fast. On Windows `torc-tune` measures both ways and keeps the faster; `TORC_PIN=1` (pin) or
   `TORC_PIN=0` (don't) overrides it. torc-tune doesn't measure it on Linux yet; `TORC_PIN=1` there has measured
   faster on a Snapdragon X under Ubuntu (a user's test).
3. ARM: `TORC_Q4X4=1` is off by default on the Qwen models and often helps. `TORC_ATTN_TILE` is on by default for the
   Qwen3.6-35B and the other Qwen3.5 / 3.6 MoE models (arch `qwen35moe`) and off for the rest; try `1` on them.
   Any machine: `TORC_ATTN_FAST=1` (fast attention) reads long prompts faster. It should stay accurate (scores within
   about 3e-5 of the largest) but is still in testing: off by default, replies can differ a little.
4. Snapdragon X NPU: `TORC_NPU=1` reads prompts faster. On one machine torc-server counts the experts in use and the NPU
   takes the most used ones from the next load on (the first session is slower than later ones; `TORC_NPU_HOT=0` turns
   that off). It is not bit-exact, so it only touches prompts, never replies. Run `npucheck` first. On one machine
   with a `qwen35moe` model held whole in memory, the NPU's defaults are blocks of 1024 and 32 hot experts a layer
   (16 without 8 GB to spare); other models keep 256 and 16.
5. GPU (in the release zips, or built with the gpu feature): `TORC_GPU_ATTN=1` shares long prompts' attention between the GPU and the CPU,
   learning the split per machine. Not bit-exact, prompts only, a gain from a few thousand tokens of prompt. If the card
   runs out of memory, the tier switches off for the rest of the run and the CPU carries on.
   `TORC_GPU_CHECK=1` compares it with the CPU; `TORC_GPU=<name part>` picks the adapter (default: the integrated
   GPU). A GPU that is no gain over the CPU alone is switched off for that model, with a line in the log, and stays
   off (kept in the tune file). Do not turn this on for desktop Intel integrated graphics: it is slower than the CPU
   and its driver can crash on some models. With the GPU on, replies can differ between runs of the same prompt.
6. Large pages (needs the "Lock pages in memory" right): `TORC_KLAY_LARGE`, `TORC_EXPERTS_LARGE`, `TORC_OWN_LARGE`.
   They can fail on a machine that has been up for days; measure.

Every `TORC_*` variable in the environment overrides a tuned value.

## 6. Bigger than RAM, or two machines

- A model larger than RAM streams its experts from the SSD through the expert cache. `TORC_CACHE_MB` sets its
  budget; [EXPERT-CACHE.md](EXPERT-CACHE.md) lists the read settings. Keep the model on the fastest internal drive.
- Two machines ([TWO-BOX.md](TWO-BOX.md)): the worker runs `torc-run <model.torc> serve-pipe=50052 ctx=<N>`, with N at least the
  main machine's context (the worker defaults to 4096), and the main machine sets `TORC_PIPE=<worker>:50052`. Both
  need the same model file. For a model that fits on one machine, use both for the prompt and `TORC_PIPE_PULL=1` to
  write the reply on one.

## 7. Measure properly

Quote numbers only from this method, so they can be compared:

    node tools/bench/torc-bench.js 8080 --model "<model and quant>" --settings "<how the server was started>"

It sends fixed texts through the server (a short prompt, then 1k, 4k, 8k and 16k tokens; `--sizes` adds 32k), warms
each size up, measures it twice and takes the server's own timings: read = prompt tokens / the time to read them,
write = reply tokens / the time to write them. One row a size goes into torc-bench.csv with the hardware filled in ([RESULTS.md](RESULTS.md)). Rules:

- Warm: send one request and discard it before measuring.
- Alternate when comparing two settings: A B B A, never A A B B.
- Enough runs: a few percent needs tens of requests a side.
- Nothing else running on the machine during a measurement, and the profiler (`TORC_PROF=1`) off.

`torc-run` and the profiler show where the time goes, but the server's numbers are the ones to quote.

Each row also keeps the settings the server ran with (`GET /torc/settings` on torc-server: the values in use, the
layout and the model file's key). `POST /torc/settings` with that profile writes its values into this machine's tune
file for the same model, used from the server's next start ([AUTOTUNE.md](AUTOTUNE.md), Sharing settings).

## 8. Troubleshooting

- **The server refuses the file:** run `torc check model.torc`. A file from an older ToRc build may need converting
  again; the error message names the layout it found.
- **Much slower than expected:** read `hwscan`'s warnings; check the drive the model is on; check the thread count
  and that nothing else is busy; with a model larger than RAM, check the cache budget.
- **Out of memory or heavy paging:** lower the context length, or let the expert cache stream the model
  (`TORC_CACHE_MB`). Budget for more than the file: a 20 GB model on a 32 GB machine with the NPU on holds about
  24 GB, 25 GB while it reads a 32k prompt. With a cache, check that the machine isn't paging once the cache is full.
- **NPU errors:** run `npucheck`; set `TORC_NPU_DIR` to the ggml build with the Hexagon backend.
- **Two machines hang or refuse:** both must run the same ToRc build and the same model file; the worker logs why it
  refused.
