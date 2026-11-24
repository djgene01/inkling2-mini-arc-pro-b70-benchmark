# Inkling 2 Mini on one Intel Arc Pro B70

A detailed, reproducible benchmark of Thinking Machines' **Inkling 2 Mini** (26.4B-parameter looped MoE, 3.4B active)
served by **vLLM on a single 32 GB Intel Arc Pro B70** with **DSpark speculative decoding**.

**[Open the full report](report/index.html)** (one self-contained HTML page: charts, tables, every number).

Measured on 24 November 2026. Raw results for every request are in [`results/`](results/), the exact serving recipe
is in [`recipe/`](recipe/), and the benchmark harness that produced everything is in [`bench/`](bench/).

## Headline results

| What | Result |
|---|---|
| Decode, single request (12 everyday prompts, low effort) | **82.7 tok/s** median (66 to 125) |
| Decode, peak (code generation in the coding run) | **203.9 tok/s** |
| Time to first token, short prompts | **171 ms** median |
| Cold prompt processing | **3,290 tok/s** peak (8K), **2,133 tok/s** at 120K |
| Cold time to first token at 120K (122,931 tokens) | **57.6 s** |
| Same prompt again (prefix cache), any size to 120K | **0.16 to 0.98 s** |
| Decode at 120K context | **59.6 tok/s** |
| 128K needle recall (32K to 124K, fact at 10 / 50 / 90 %) | **11 of 12** (see the 64K finding below) |
| Coding run (agent with tools, real test execution) | **PASS**: 10 tests written and passing, 8 turns, 20 s |
| DSpark, all single-request tests | **29.0 %** of drafted tokens kept, **3.03** tokens per verify step |
| DSpark in the coding run | **66.5 %** kept, **5.66** tokens per step |
| Concurrent, 8 short chats | **304 tok/s** aggregate (44.5 tok/s each), first token 218 ms |
| Energy (whole board) | **2.9 J/token** single stream, **0.76 J/token** at 8 streams |

### Long context, cold and cached

| Context | Prompt tokens | Cold first token | Prefill tok/s | Cached first token | Decode tok/s |
|---|---:|---:|---:|---:|---:|
| 1K | 1,083 | 0.40 s | 2,725 | 0.16 s | 80.2 |
| 4K | 4,155 | 1.30 s | 3,207 | 0.18 s | 76.4 |
| 8K | 8,250 | 2.51 s | 3,290 | 0.18 s | 83.0 |
| 16K | 16,441 | 5.13 s | 3,205 | 0.20 s | 69.6 |
| 32K | 32,825 | 10.97 s | 2,991 | 0.25 s | 67.8 |
| 64K | 65,584 | 25.75 s | 2,547 | 0.98 s | 63.4 |
| 96K | 98,349 | 42.40 s | 2,319 | 0.54 s | 64.3 |
| 120K | 122,931 | 57.64 s | 2,133 | 0.51 s | 59.6 |

### Concurrency

| Load | Aggregate tok/s | Per stream tok/s | Median first token | J/token |
|---|---:|---:|---:|---:|
| 1 short chat | 90.6 | 93.8 | 164 ms | 2.52 |
| 2 short chats | 153.2 | 79.0 | 164 ms | 1.50 |
| 4 short chats | 196.1 | 64.6 | 179 ms | 1.17 |
| 8 short chats | 304.0 | 44.5 | 218 ms | 0.76 |
| 4 chats with 8K-token prompts | 109.0 | 40.0 | 8.02 s | 2.11 |

### Finding: a narrow recall miss just past 64K

Recall was perfect at 32K, 96K and 124K. The one miss was in the 64K case. Probing it further: prompts of 63K, 65.5K
and 70K tokens answered every time, but prompts of about **65,620 tokens** missed in 5 of 7 tries. At that length the
model describes the pasted code and says no question was asked, as if it does not see the question at the very end.
It looks like an edge case in the serving stack at that exact length, not a general long-context limit. Prompts are
prefilled in 2,048-token chunks, and the last chunk there is only about 84 tokens. That is a hypothesis, not tested.
All probes are in `results/res-recall-64k-*.json`.

## Test system

| | |
|---|---|
| GPU | Intel Arc Pro B70, 32 GB GDDR6 (Battlemage BMG-G31, Xe2), single card, PCIe 4.0 x16 |
| GPU power limit | 230 W (board), max GPU clock 2,800 MHz |
| CPU | AMD Ryzen 7 3700X (8 cores, 16 threads) |
| Memory | 60 GB usable system RAM (as reported by the OS) |
| Motherboard | MSI MPG B550I GAMING EDGE WIFI |
| OS | Ubuntu 26.04 LTS, Linux 7.0.0-44-generic, `xe` kernel driver |
| Host GPU runtime | Intel compute runtime (`libze-intel-gpu1`, `intel-opencl-icd`) 26.05.37020.3 |
| Container runtime | Docker 29.1.3 |
| In the container | Ubuntu 24.04 (Intel oneAPI toolkit image), compute runtime 26.18.38308.1, Level Zero loader 1.28.2 |
| Python / PyTorch | Python 3.14 (uv venv), PyTorch 2.15.0+xpu, Triton 3.9.0+xpu, vllm-xpu-kernels 0.1.17 |
| vLLM | 0.34.0rc1.dev1133+g647a76866 (XPU wheel) |
| Model | [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) @ `ed40e7da`, native NVFP4 / MXFP8 experts, unmodified |

### Serving settings

128K context (131,072), FP8 KV cache, prefix caching on, DSpark with 7 draft tokens, max 2,048 batched tokens, GPU
memory utilisation 0.92, block size 64, decode graphs only (`-cc.cudagraph_mode=FULL_DECODE_ONLY`), vision on,
`inkling` reasoning and tool-call parsers, one request at a time (`--max-num-seqs 1`) except the concurrency test (8).
Both opt-in B70 kernels on: `INK2_CUTLASS_MOE=1` (AOT `bmg_g31` NVFP4 CUTLASS Xe2 MoE) and `INK2_SYCL_DECODE_ATTN=1`.

## What is in this repo

```
report/index.html            the full report (self-contained; uses Google Fonts if online)
results/res-*.json           raw results: every request with timings, tokens, DSpark counters and energy
recipe/Dockerfile            the image used, with the build fixes described below
recipe/launch.sh             the exact docker run / vllm serve command
recipe/b70-control-recipe.json   the same settings as a B70 Control recipe
recipe/patches/              SergiioB's Inkling-on-B70 patches (MIT, see LICENSE-SergiioB-cookbook-MIT)
bench/bench.py               the benchmark harness (phases: env warmup burst ladder recall coding concurrency)
bench/gen_report.py          builds report/index.html from results/
```

## Reproduce

1. Download the model at revision `ed40e7da` and check `model-00001-of-00001.safetensors` against sha256
   `65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1`.
2. Build the image: `cd recipe && docker build --network host -t b70-inkling-vllm:2026-11-23 .`
3. Start it: `MODEL_DIR=/path/to/Inkling-2-Mini ./recipe/launch.sh` (about 3 minutes to load).
4. Run the phases, then build the report:
   ```
   cd bench
   for p in env warmup burst ladder recall coding; do python3 bench.py $p ../results/res-$p.json; done
   # restart with MAX_SEQS=8 ./recipe/launch.sh, run concurrency twice and keep the second:
   python3 bench.py concurrency /tmp/warm.json && python3 bench.py concurrency ../results/res-concurrency.json
   python3 gen_report.py ../results ../report/index.html
   ```
   The harness needs Docker for the coding test (the model's code runs offline in `python:3.12-slim`, set
   `SANDBOX_IMAGE` to change it) and reads GPU energy from `/sys/class/hwmon/hwmon5/energy1_input` (adjust for your
   machine; energy columns are left empty if it is not readable).

## How it was measured

- Requests go straight to vLLM's OpenAI API, streamed, with usage returned at the end.
- **First token**: the first streamed token of any kind, including reasoning. **First answer**: the first visible reply token.
- **Prefill** = uncached prompt tokens / time to first token. **Decode** = (output tokens - 1) / (end - first token).
- **DSpark**: deltas of vLLM's `spec_decode` counters around each request. Kept = accepted / drafted;
  tokens per step = 1 + accepted / drafts.
- **Energy**: the card's own hwmon energy counter (whole board) over each request.
- Every prompt size was run once before measuring, so first-use kernel compilation is not in the results. Long prompts
  are real CPython standard-library source, cut to exact token counts with vLLM's `/tokenize`, with a unique first line
  so cold runs never hit the prefix cache.
- vLLM's request counter matched the harness in every phase (24, 20, 12, 8, 19), so no other traffic reached the model.
- Sampling: temperature 0.6 (0.3 for the coding run), top-p 0.95, reasoning effort low unless stated.
- Recall history: a first version that called the fact a secret "access code" made the model decline to repeat it; it
  was changed to a harmless release tag. `res-recall-run1.json` is the run before an end-of-files marker was added.

## Build notes (fixes over the published route)

Following SergiioB's cookbook route on a stock Intel oneAPI image needed these changes, all in `recipe/Dockerfile`:

- PyTorch 2.15+xpu segfaults on its first device query under compute runtime 26.40 (pulled in with `ocloc`), so
  `ocloc` lives only in a build stage and the final image keeps the image's 26.18 runtime.
- The oneAPI image's `LD_LIBRARY_PATH` makes PyTorch load the toolkit's older `libur_loader`; the venv's libraries go
  first for the kernel build and the variable is cleared at run time.
- The NVFP4 CUTLASS build needed extra include folders (vllm-xpu-kernels root, sycl-tla `tools/util/include` and
  `applications`), the SPIR-V extensions vllm-xpu-kernels passes (`split_barrier`, `2d_block_io`,
  `subgroup_matrix_multiply_accumulate`, `-flink-huge-device-code`) for icpx 2026.0, and
  `-DTORCH_EXTENSION_NAME=_inkling_nvfp4_xe2`, without which Python cannot import the module.
- The last hunk of `vllm-inkling2-xpu-cutlass-optin.patch` declares 14 added lines but has 13.

## Credits and licences

- Model: Thinking Machines, Apache-2.0.
- XPU support: [mgaruccio/vllm-inkling2-xpu](https://github.com/mgaruccio/vllm-inkling2-xpu) (Apache-2.0) and
  vLLM [PR #69186](https://github.com/vllm-project/vllm/pull/69186).
- B70 route, kernels and patches: [SergiioB/intel-arc-pro-b70-inference-cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook)
  (MIT); the copies in `recipe/patches/` keep that licence.
- Everything else in this repo: MIT.

Independent community benchmark. Not produced or endorsed by Intel or Thinking Machines. Intel and Arc are trademarks
of Intel Corporation.
