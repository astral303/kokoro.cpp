# GPU Inference Optimisation Plan

## Purpose

This document defines the target architecture, known correctness hazards, and
repeatable validation loop for improving Kokoro GPU inference. GPU work must be
incremental: every optimisation needs a fixed-input correctness report before
it is accepted, and performance measurements must never substitute for output
quality checks.

The immediate objective is not maximum throughput. It is a trustworthy GPU
implementation that can be changed, measured, and debugged without relying on
subjective listening alone.

## Target architecture

The ideal steady-state utterance path is:

```text
CPU preprocessing
  text -> phonemes -> token IDs -> small dynamic-shape decisions

One large device-resident compute region
  persistent model weights
  native GGML backend kernels
  device-resident intermediate tensors
  few scheduler submissions and synchronisation points

CPU postprocessing
  final device download -> waveform packaging
```

A small number of graphs is acceptable when a data-dependent shape requires a
host decision. The important properties are:

1. Model weights remain resident on the selected device.
2. Large intermediate tensors do not round-trip through host memory.
3. Graphs are functional DAGs. Correctness must not depend on hidden mutation,
   execution order, or side effects through tensor views.
4. Backend-specific reduced precision is explicit and validated.
5. Unsupported operations fall back deliberately, not accidentally.
6. CPU, Vulkan, Metal, and CUDA use the same public API and validation gates.

## Current pipeline

The current implementation still uses stage boundaries that are convenient for
CPU debugging, but duration/text alignment no longer expands on the host, and
BERT now feeds the duration predictor inside one functional graph:

```text
1.  BERT + duration predictor graph
      padded token IDs / positions in
      bert_proj_out stays device-resident into the predictor
      download duration encoding and duration values
2.  CPU duration rounding
      build int32 alignment indices of length T_frames
3.  F0/noise graph
      upload compact dur_enc (640,L) + indices; device ggml_get_rows → en
      download F0 and noise curves
4.  Vulkan only: launch CPU harmonic source asynchronously
      host-cached m_source weights + f0_curve + fixed-seed RNG
      overlaps stages 5–6; joined before the generator
5.  Text encoder graph
      download text encoding
6.  Decoder graph
      upload compact text_enc (512,L) + same indices; device gather → asr
      download decoder activation
7.  Join harmonic source (no-op wait when fully overlapped)
      upload decoder activation and harmonics
8.  Generator graph
      download magnitude and phase
9.  CPU iSTFT
      return PCM
```

Host-side alignment used to materialise and upload `(640,T_frames)` and
`(512,T_frames)` tensors. Those expansions are now device gathers, so the H2D
payload for alignment is approximately `(640+512)·L` floats plus `2·T_frames`
int32 indices instead of `(640+512)·T_frames` floats. Merging BERT with the
duration predictor also removed the `bert_proj_out` D2H host vector, its H2D
re-upload, one scheduler reset/allocation, and one graph submission. The
duration vector remains the legitimate host synchronisation point for dynamic
`T_frames`. On Vulkan, harmonic generation no longer waits for TextEncoder +
Decoder: it starts immediately after F0/N using only host-cached `m_source`
weights (copied at context init — never `ggml_backend_tensor_get` during
overlap), immutable `f0_curve`, and the fixed-seed RNG. CPU/Metal/CUDA keep
the historical serial ordering. Scheduler resets between later graphs still
prevent keeping predictor/text-encoder allocations resident across stages.

This structure was originally reasonable for CPU-only inference: a backend
upload/download was generally a host-memory copy, and independent stages were
easier to validate. On a discrete GPU, the remaining round-trips still introduce
PCIe transfers, scheduler resets, device synchronisation, and many small graph
submissions.

## Confirmed findings

### 1. Fixed-seed CPU/GPU differences were not random

`KOKORO_SEED=1234` produced bit-reproducible output on Vulkan. CPU output was
also bit-reproducible. The original low waveform correlation therefore was not
caused by random source generation or nondeterministic dispatch.

### 2. Side-effecting writes through tensor views broke Vulkan graph optimisation

The original LSTM helper allocated an output tensor and populated each column
with this pattern:

```text
view(output, column t) <- cpy(h_t)
```

The copy nodes were separately expanded into the graph, while downstream nodes
read `output`. The data dependency existed only as mutation through aliased
views; it was not represented as a functional edge from each `h_t` to the
consumer.

CPU execution happened to preserve the expected order. The Vulkan graph
optimiser transformed or scheduled the graph without preserving that implicit
side-effect ordering. Intermediate comparison isolated the discontinuity:

| F16 stage | Relative RMS difference before correction |
|---|---:|
| Text encoder CNN output | `2.52e-5` |
| Text encoder LSTM output | `1.82e-1` |

Disabling Vulkan graph optimisation made the LSTM discrepancy disappear. The
permanent correction replaced copy-to-view output assembly with a balanced,
functional concat tree. This preserved CPU output bit-for-bit and made the
LSTM graph backend-safe without changing process-global Vulkan settings.

**Rule:** persistent cache mutation may require explicit backend-specific
handling, but ordinary feed-forward and recurrent graph outputs must not use
hidden copy-to-view side effects.

### 3. Backend precision differences are amplified by recurrent and phase stages

Vulkan can select FP16 accumulation and backend-specific quantised matrix
kernels. Kokoro contains recurrent networks followed by magnitude/phase
prediction and iSTFT. Small upstream errors can affect recurrent state and
phase; phase differences are then amplified in time-domain PCM.

Kokoro now requests `GGML_PREC_F32` for its matrix multiplications. This
improved Vulkan parity, but it does not govern every convolution or quantised
backend kernel. In particular, Q8_0 still differs more than F16.

Current fixed-seed end-to-end baseline after the LSTM correction and F32 matmul
hints, using an AMD Radeon RX 7900 XTX and upstream GGML 0.17.0:

| Model | Samples | CPU/Vulkan correlation | Relative RMS delta | ASR result |
|---|---:|---:|---:|---|
| F16 | 61,800 | `0.9535` | `0.3045` | identical transcript |
| Q8_0 | 61,800 | `0.6907` | `0.7853` | identical transcript |

These numbers are diagnostics, not acceptance thresholds. F16 intermediate
activations are much closer than final PCM (for example decoder correlation
was above `0.99998` during stage analysis), demonstrating the sensitivity of
phase reconstruction. Q8_0 does **not** yet meet a strict numerical-parity
claim and needs backend-kernel investigation.

### 4. Dynamic durations create a real host synchronisation boundary

The duration predictor emits floating-point durations. The host rounds and
clamps them, computes `T_frames`, and only then can build later dynamic-shape
graphs. Reading this small duration vector is a legitimate synchronisation
point.

Alignment no longer downloads for the sake of host expansion: after the
duration indices are known, F0/N and decoder graphs gather compact `dur_enc` /
`text_enc` with a host-generated int32 index tensor via `ggml_get_rows`. The
encodings are still downloaded today only because scheduler resets discard the
producer allocations; keeping them device-resident across resets is a later
step.

### 5. CPU custom stages force additional boundaries

The following stages currently use ordinary C++ rather than GGML graphs:

- harmonic source and random-noise generation;
- final iSTFT.

Duration/text alignment indices are still built on the host, but the feature
expansion itself is a device gather. On Vulkan the harmonic source now overlaps
TextEncoder + Decoder and only joins before the generator upload; the
magnitude/phase download remains a real boundary. Moving the remaining CPU
stages onto the device must be done only after native-kernel parity tests exist.

### 6. Scheduler fallback can hide data movement

The scheduler contains a primary backend and CPU fallback. This is useful for
unsupported operations, but a successful graph does not prove it stayed on the
GPU. Validation should record backend assignment and transfer counts once GGML
exposes or Kokoro adds suitable diagnostics.

### 7. Shared GGML process state belongs to the host

Kokoro may be embedded in a process that shares one GGML library with other
models. Kokoro must not modify process-global GGML logging, environment, or
backend policy. `kokoro_log_set()` controls only Kokoro diagnostics; the host
owns GGML logging. The same rule applies to proposed workarounds such as
setting `GGML_VK_DISABLE_F16` from library code: experiments may set it in the
launcher, but the library must not mutate the process environment.

Backend controls need an explicit ownership/lifetime classification:

| Control class | Examples | Owner and lifetime |
|---|---|---|
| Build capability | `KOKORO_VULKAN`, `KOKORO_CUDA`, external `ggml` target | Build system; determines which backend symbols/implementations exist |
| Kokoro context policy | backend, device, thread count; future stable optimisation flags | `kokoro_params`; fixed when a context is created and independent between contexts where GGML permits |
| Kokoro experimental/debug control | `KOKORO_HAR_OVERLAP`, `KOKORO_LSTM_BATCH`, `KOKORO_BENCH` | Kokoro/launcher; should migrate to typed context options if promoted to supported API |
| GGML backend policy | `GGML_VK_ALLOW_GRAPHICS_QUEUE`, `GGML_VK_DISABLE_F16`, Vulkan submit/queue variables | Embedding host; process-global and generally fixed by the first backend initialisation |

An extensive Kokoro knob API should expose Kokoro-owned graph/scheduling
choices through typed init options rather than setting environment variables.
GGML-owned variables must remain host/launcher controls: copying them into
`kokoro_params` would falsely imply per-context scope and would require Kokoro
to mutate shared process state. The standalone CLI may offer convenience flags
that set GGML variables before it calls any GGML/Kokoro API because the CLI
owns that process; an embedding library API must not do so.

## Correctness gates

Optimisations are evaluated in layers. A later gate does not replace an earlier
one.

### Gate 0: build and API

- CPU-only and requested GPU configurations build with warnings-as-errors.
- Model-free C and C++ tests pass.
- Explicit unavailable backends fail cleanly.
- No Kokoro-owned library code writes to stdout or stderr (CLI diagnostics are
  allowed).

### Gate 1: deterministic inference invariants

With fixed model, voice, text, backend, device, and seed:

- repeated runs are bit-reproducible unless a documented backend limitation
  says otherwise;
- output is finite, mono, and at the model sample rate;
- CPU and candidate produce the same sample count and duration;
- model and voice hashes are recorded.

### Gate 2: numerical diagnostics

Record, but do not blindly gate on:

- RMS and maximum absolute PCM delta;
- relative RMS delta and signal-to-difference ratio;
- waveform correlation;
- eventually, per-stage RMS/max/correlation.

PCM correlation is especially sensitive to phase and must be interpreted with
intermediate metrics. Conversely, matching transcripts must not excuse a large
unexpected discontinuity in an early model stage.

### Gate 3: semantic quality

- Transcribe CPU and candidate WAVs with the same pinned ASR model.
- Require equal normalised transcripts for the basic regression gate.
- Keep generated WAVs and JSON reports for listening when a numerical metric
  moves materially.
- Add a small fixed corpus covering short/long text, punctuation, numbers,
  phoneme input, and multiple voices/languages as those assets become
  validated.

### Gate 4: performance

Measure only after correctness artifacts are produced:

- end-to-end process time (load + synthesis + WAV write);
- eventually context-load time and warm in-process synthesis time separately;
- median and range across repeated runs;
- peak host RSS and device memory;
- graph submissions, CPU fallback nodes, and transfer bytes when available.

## Experiment protocol

Every experiment should:

1. Start from a clean commit.
2. Change one variable or clearly defined group of variables.
3. Use fixed text and `KOKORO_SEED`.
4. Record model, voice, executable, GGML revision, backend, device, and relevant
   `KOKORO_*`/`GGML_*` environment variables.
5. Run CPU reference and candidate through the same executable.
6. Save WAVs and a machine-readable JSON report.
7. Run optional ASR comparison.
8. Report correctness before timing.
9. Revert experiments that improve speed by violating an earlier gate.

## Staged roadmap

### Phase A — repeatable end-to-end loop

- Build CPU/Vulkan configurations reproducibly.
- Generate matched fixed-seed WAVs.
- Calculate deterministic, format, numerical, timing, and optional ASR metrics.
- Preserve JSON reports and artifacts.

The scripts added with this document provide this baseline.

## Validation tools

Run commands from the repository root. In environments where espeak-ng and the
GPU SDK are supplied by Nix, enter the development shell first:

```bash
nix develop
cd third_party/kokoro.cpp
```

### Rapid inner loop

```bash
tools/run_backend_validation.sh --backend vulkan --quick
```

Quick mode:

- uses the short text `Hello.`;
- tests F16 by default (pass `--model` to target Q8_0);
- performs one candidate run;
- creates a CPU reference on first use and reuses it thereafter;
- still compares WAV format, sample count, and all PCM metrics;
- rebuilds incrementally unless `--skip-build` is supplied.

On the RX 7900 XTX, candidate synthesis for the current quick fixture takes
about 0.4 seconds after the build is current. The initial run is slower because
it generates the CPU golden WAV.

The CPU reference intentionally survives source changes so GPU-only experiments
can get rapid feedback. Regenerate it after any change that may affect CPU
output, model/voice assets, synthesis parameters, or fixture semantics:

```bash
tools/run_backend_validation.sh --backend vulkan --quick --refresh-reference
```

Quick mode is a smoke/regression loop, not the final acceptance gate. One run
cannot establish repeated-run determinism, and F16 alone does not cover
quantised kernels.

### Full pre-commit verification

```bash
tools/run_backend_validation.sh --backend vulkan
```

The full default builds with `KOKORO_WERROR=ON`, runs model-free C/C++/Python
tests, then performs two CPU and two candidate runs for both F16 and Q8_0.
Pass `--full-corpus` for the longer text corpus. Important controls include
`--backend`, `--device`, repeated `--model`, `--voice`, `--threads`, `--seed`,
`--runs`, `--warmups`, `--build-dir`, and `--output-dir`.

For semantic validation, point the script at an already running
OpenAI-compatible transcription service:

```bash
tools/run_backend_validation.sh \
    --backend vulkan \
    --transcription-url http://127.0.0.1:18001/v1/audio/transcriptions
```

The ASR request uses the `file`, `model`, `language`, and `response_format`
multipart fields. CPU and candidate transcripts are normalised for case,
punctuation, and whitespace before comparison.

For a single controlled experiment or custom numerical threshold, invoke the
Python tool directly:

```bash
python3 tools/backend_validation.py \
    --cli .validation/build-vulkan/kokoro-cli \
    --model models/kokoro-82m-f16.gguf \
    --voice models/kokoro-voice-af_heart.gguf \
    --backend vulkan --device 0 \
    --seed 1234 --runs 2 \
    --min-correlation 0.95 \
    --output-dir .validation/manual-f16
```

Generated WAVs, stderr logs, hardware summaries, cached references, and JSON
reports live below `.validation/` and are ignored by Git. The JSON report
records executable/model/voice hashes, revisions, relevant environment,
configuration, timings, determinism, invariants, numerical metrics, ASR output,
and gate failures.

### Recorded Vulkan baseline

The full harness reproduced the previously measured corrected outputs on
2026-07-21:

| Model | Correlation | Relative RMS delta | Median CPU wall | Median Vulkan wall |
|---|---:|---:|---:|---:|
| F16 | `0.953535` | `0.304508` | 8.869 s | 0.299 s |
| Q8_0 | `0.690742` | `0.785317` | 8.562 s | 0.301 s |

These are end-to-end process times (model load, one synthesis, and WAV write),
measured with four CPU threads on the hardware described above. They are not
steady-state library benchmarks.

With `KOKORO_BENCH=1`, the CLI installs a Kokoro log callback and reports
synchronised pipeline-stage times. Five F16 runs of the baseline fixture on the
RX 7900 XTX gave these representative medians **before** merging BERT into the
duration predictor and **before** `GGML_VK_ALLOW_GRAPHICS_QUEUE` (default
compute queue). Do not compare these stage numbers directly to later
graphics-queue audit tables:

| Stage | Median |
|---|---:|
| BERT | 7.21 ms |
| Predictor | 22.40 ms |
| F0/noise | 15.27 ms |
| Text encoder | 5.72 ms |
| Decoder | 8.18 ms |
| CPU harmonic source | 9.02 ms |
| Generator | 39.32 ms |
| CPU iSTFT | 1.08 ms |
| Complete synthesis pipeline | 107.91 ms |

The timer for each graph includes graph construction, scheduler allocation,
execution, and its immediately required transfers. It excludes model loading,
phonemisation, and WAV writing. This makes graph-merging experiments visible
without conflating them with process startup.

Run a directly inspectable profile with:

```bash
KOKORO_BENCH=1 KOKORO_SEED=1234 \
  .validation/build-vulkan/kokoro-cli \
  -m models/kokoro-82m-f16.gguf \
  -v models/kokoro-voice-af_heart.gguf \
  --backend vulkan --device 0 -o .validation/profile.wav \
  "The documented Q Eight model works."
```

After replacing host duration/text expansion with device `ggml_get_rows`
gathers (same hardware, seed, and baseline text), one-run F16/Q8_0 validation
reproduced the same correlations and relative RMS deltas (`0.953535` /
`0.304508` and `0.690742` / `0.785317`). Quick-mode CPU PCM remained
bit-identical to the prior cached golden.

After merging BERT and the duration predictor into one functional graph (same
hardware, seed, baseline text, graph optimiser still enabled), one-run F16/Q8_0
validation again reproduced those exact correlations and relative RMS deltas.
Six `KOKORO_BENCH=1` F16 runs measured:

| Stage | Pre-merge median | Post-merge median |
|---|---:|---:|
| BERT + Predictor | 29.61 ms (7.21 + 22.40) | 27.11 ms (`bert_predictor`) |
| Complete synthesis pipeline | 107.91 ms | 103.77 ms |

The combined stage is about 2.5 ms faster than the sum of the previous separate
stages, and the pipeline total drops by about 4.1 ms. Treat that as end-to-end
stage-timer improvement from removing one host round-trip plus one scheduler
reset/submission — not as a claim that BERT/predictor FLOPs themselves got
faster.

### Generator time-first layout (Vulkan)

The generator historically kept large activations in channel-first `(C, T)` and
paid a contiguous transpose around every AdaIN (`ggml_norm` wants `(T, C)`) and
every `ggml_conv_1d` (also `(T, C)`). Snake-α used `DIV` with broadcast
`alpha=(C, 1)`, which cannot match Vulkan's fused Snake kernel.

Vulkan Snake fusion requires the exact chain `MUL → SIN → SQR → MUL → ADD`,
contiguous `x=(T, C)`, and F32 `a` / `inv_a` shaped `(1, C)`. A layout rewrite
is therefore what makes both CONT-copy removal and Snake fusion possible; merely
replacing `DIV` with `MUL` on the old `(C, T)` path is not enough.

On `KOKORO_BACKEND_VULKAN` only, the generator now:

- converts `x_in` / `har` from `(C, T)` to `(T, C)` once at the graph boundary;
- runs AdaIN via `ggml_norm` directly on `(T, C)` with `(1, C)` gamma/beta;
- keeps Conv1d residual paths in `(T, C)` without per-op transpose CONT;
- emits Snake as `MUL→SIN→SQR→MUL→ADD` with graph-computed `(1, C)` reciprocal
  (reciprocal nodes are expanded before the Snake chain so they do not break
  fusion contiguity);
- still converts at each decomposed ConvTranspose boundary via
  `convt1d_decomp_tf` (GGML itself is unchanged);
- converts back to channel-first only for the named `gen_pre_post_out` / mag /
  phase outputs expected by CPU iSTFT.

CPU keeps the historical channel-first builder and remains bit-identical to the
cached golden. Metal/CUDA are unchanged for now.

One-run F16/Q8_0 validation after this change (same hardware, seed, baseline
text):

| Model | Correlation | Relative RMS delta |
|---|---:|---:|
| F16 | `0.953395` | `0.304968` |
| Q8_0 | `0.690435` | `0.785729` |

These are small, deterministic shifts from the prior `0.953535` / `0.304508`
and `0.690742` / `0.785317` baselines. They are expected because fused Snake
changes floating-point evaluation order, and they are not being described as
run-to-run noise. The first changed region is deliberately confined to the
generator; earlier graphs are unchanged. Quick-mode CPU PCM stayed
bit-identical to the cached reference.

Six `KOKORO_BENCH=1` F16 runs on the RX 7900 XTX after the layout change:

| Stage | Post-merge median | Time-first median |
|---|---:|---:|
| Generator | ~37–39 ms | 28.46 ms |
| Complete synthesis pipeline | ~102–104 ms | 94.00 ms |

Generator improved by about 25% versus the post-merge ~38 ms level. A
`GGML_VK_PERF_LOGGER=1` capture of the two main generator splits showed fused
`SNAKE MUL: 24` each (48 total), CONT reduced from ~100+102 to 4+4, snake
`SIN`/`SQR`/`DIV` removed from those splits (one remaining `SIN` is phase), and
reciprocal `DIV` isolated to a small prep split (`DIV: 48`).

### Harmonic / TextEncoder+Decoder overlap (Vulkan)

`kokoro_make_har` depends only on `f0_curve`, `T_frames`, the immutable
`m_source` Linear(9→1) weights, and the fixed-seed RNG — not on text-encoder
or decoder outputs. Those tiny weights are copied into host fields at context
init (F16 or F32) so the worker never calls `ggml_backend_tensor_get` while
the Vulkan scheduler is computing. On `KOKORO_BACKEND_VULKAN` only, harmonic
generation launches immediately after F0/N and joins before the generator.
CPU/Metal/CUDA keep the serial order. `KOKORO_HAR_OVERLAP=0` disables the
overlap for A/B measurement.

Worker elapsed time is recorded on the worker and logged on the caller thread
as `harmonic_source`; `harmonic_wait` is the residual join after TextEncoder +
Decoder. `pipeline_total` naturally reflects the overlapped wall time. RAII
(`core_har::AsyncResult`) joins and frees on every early-return path.

Correctness after this change (same hardware, seed, baseline text; graph math
unchanged):

| Model | Correlation | Relative RMS delta |
|---|---:|---:|
| F16 | `0.953395` | `0.304968` |
| Q8_0 | `0.690435` | `0.785729` |

These match the time-first baseline exactly. Quick-mode CPU PCM stayed
bit-identical to the cached golden. Twelve interleaved
`KOKORO_HAR_OVERLAP=0/1` pairs produced bit-identical Vulkan WAVs.

Twelve interleaved `KOKORO_BENCH=1` F16 pairs on the RX 7900 XTX:

| Stage | Serial median | Overlap median |
|---|---:|---:|
| Harmonic source (worker) | 8.86 ms | 9.93 ms |
| Harmonic wait (join) | — | 0.00 ms |
| Text encoder + decoder | 13.04 ms | 13.15 ms |
| Generator | 28.27 ms | 21.16 ms |
| Complete synthesis pipeline | 93.25 ms | 77.00 ms |

Pipeline median improved by 16.25 ms (mean 16.91 ms; every paired run ≥13.45
ms). The harmonic wait is effectively zero, so the ~9 ms CPU harmonic work is
fully hidden behind TextEncoder + Decoder. The generator also ran ~7 ms faster
in the overlapped configuration under the same interleaved protocol; treat
that as an observed scheduling effect, not as a change to generator graph math:
removing the CPU-only gap keeps the Vulkan queue busy immediately before the
generator and likely avoids device clock/queue warm-up loss.

### Bidirectional LSTM direction batching (Vulkan)

Every Kokoro bidirectional LSTM previously ran `lstm_unidir` twice (forward,
then reverse), duplicating recurrent `mul_mat` and gate elementwise dispatches
per timestep. On `KOKORO_BACKEND_VULKAN` only, both directions now share GGML
batch dimension `ne[2]=2`:

- stack forward `X` with a time-reversed gather of `X` (I32 indices from
  `arange` → negate/offset → cast);
- graph-concat forward/reverse `W_ih` / `W_hh` / biases on `ne[2]` (no
  persistent prepack);
- project once to `(4H, T, 2)` and recur on `h`/`c` shaped `(H, 1, 2)` so each
  timestep's matmul and gate ops cover both directions;
- assemble `(2H, T)` functionally: forward at `t`, reverse at `T-1-t` via the
  same reverse gather (no cpy-to-view mutation).

CPU/Metal/CUDA keep the historical dual-unidir path. `KOKORO_LSTM_BATCH=0`
disables the batched path for A/B measurement. F32 matmul precision hints are
unchanged.

Correctness after this change (same hardware, seed, baseline text; graph
optimiser enabled):

| Model | Correlation | Relative RMS delta |
|---|---:|---:|
| F16 | `0.953395` | `0.304968` |
| Q8_0 | `0.690435` | `0.785729` |

These match the post-overlap baseline exactly. Quick-mode CPU PCM stayed
bit-identical to the cached golden. Twelve interleaved
`KOKORO_LSTM_BATCH=0/1` Vulkan pairs produced bit-identical WAVs. A CPU unit
comparison of dual-unidir vs batched biLSTM also matched bit-for-bit.

Twelve interleaved `KOKORO_BENCH=1` F16 pairs on the RX 7900 XTX:

| Stage | Dual-unidir median | Batched median |
|---|---:|---:|
| `bert_predictor` | 27.05 ms | 19.86 ms |
| `f0_noise` | 13.86 ms | 10.43 ms |
| `text_encoder` | 5.22 ms | 3.64 ms |
| Complete synthesis pipeline | 75.44 ms | 64.58 ms |

Pipeline median improved by 10.86 ms (paired median delta −11.33 ms; every
paired run ≤ −9.28 ms). `GGML_VK_PERF_LOGGER=1` op-count changes for the same
fixture:

| Op | Dual-unidir | Batched | Delta |
|---|---:|---:|---:|
| `MUL_MAT_VEC*` (matvec) | 721 | 399 | −322 |
| `MUL` | 2066 | 1088 | −978 |
| `SIGMOID` | 1957 | 979 | −978 |
| `TANH` | 1312 | 656 | −656 |
| `CONCAT` | 674 | 382 | −292 |

## Cumulative result and optimisation policy

Optimisation priority follows Amdahl's law: improve the largest measured region
or remove a boundary that blocks overlap with it. Reducing a stage that is 10%
of total runtime to zero can save at most 10%; reducing its kernel count without
moving end-to-end latency is not sufficient reason to keep a complex rewrite.

Representative F16 pipeline medians on the RX 7900 XTX evolved as follows.
**This snapshot is pre-`GGML_VK_ALLOW_GRAPHICS_QUEUE`** (default compute-queue
selection). The follow-up audit below then measures the graphics-capable queue
(~59 ms F16 median on the same fixture) as a separate host-configured step.

| State | Pipeline median | Change from initial |
|---|---:|---:|
| Corrected functional LSTM baseline | 107.91 ms | — |
| BERT + predictor graph merge | 103.77 ms | −3.8% |
| Time-first Vulkan generator + fused Snake | 94.00 ms | −12.9% |
| Overlap CPU harmonic work with GPU text/decoder | 77.00 ms | −28.6% |
| Batch bidirectional LSTM directions | 64.58 ms | **−40.2%** |

These rows come from separate interleaved experiments rather than one synthetic
sum of kernel times. The final number is steady-state synthesis-pipeline time;
CLI model loading and WAV writing are excluded.

At this pre-graphics-queue point the largest remaining regions are approximately:

- BERT + duration predictor: 20 ms;
- generator: 21 ms;
- F0/noise: 10–11 ms;
- decoder: 7–8 ms;
- text encoder: 3–4 ms.

(With the graphics-capable queue those recurrent stages drop further — see the
queue A/B stage table — while generator/decoder can move slightly the other way.)

The generator's large convolutions already reach high throughput in Vulkan
`mul_mat` kernels. The recurrent regions remain launch-heavy even after
direction batching: their timestep dependency prevents batching across time,
and there is no native GGML LSTM operation.

## Follow-up no-kernel-patch audit

A follow-up audit on the same RX 7900 XTX looked specifically for improvements
that use existing GGML/Vulkan behaviour and operations without changing GGML
operators or shaders. It found one material host-configured improvement, ruled
out one plausible Vulkan tuning direction, and identified several smaller
application-graph experiments. Measurements in this section used Kokoro commit
`df4f978` and GGML commit `9be31331` on 2026-07-21.

Unless noted otherwise, every rejected experiment below was gated by a temporary
`KOKORO_*` selector that **no longer exists in source**. Metric tables and
fixture notes are retained as historical evidence; those selector values are not
runnable on the current tree. Current intended instrumentation is only
`KOKORO_BENCH` (including lifecycle / `har_phase.*` output) and CLI
`KOKORO_BENCH_ITERS`.

### RADV graphics-capable queue selection — confirmed improvement

GGML Vulkan normally prefers a compute queue family that does not advertise
graphics support. Upstream GGML also exposes
`GGML_VK_ALLOW_GRAPHICS_QUEUE=1`, with a source comment noting that allowing
the graphics-capable queue can improve RADV performance.

This is a **runtime GGML implementation feature, not a Kokoro/GGML compile-time
option**. The Vulkan code calls `getenv("GGML_VK_ALLOW_GRAPHICS_QUEUE")` while
the process-global Vulkan instance/device and queue families are first created.
The check is presence-based: any defined value, including `0`, enables it;
disable it by leaving/unsetting the variable. Once any code in the process has
caused `ggml_vk_instance_init()` (for example through Vulkan backend
registration, device enumeration, or backend creation), changing the variable
is too late and cannot reselect queues.

Linkage does not change those runtime semantics:

- the bundled GGML baseline (`9be31331`, GGML 0.17.0) supports the variable;
- the override was introduced upstream by GGML commit `16dca8ea`;
- a compatible external/static/shared GGML implementation that contains that
  code reads the same process environment at runtime;
- an older or divergent external GGML that lacks the implementation silently
  ignores the variable—there is no public GGML capability query for this
  environment knob;
- if another model or component already initialised the shared external GGML
  Vulkan backend, Kokoro must accept the queue policy already chosen.

Therefore the host/launcher must set it before any shared GGML Vulkan API is
used. Kokoro library code must not set it. Reproducible reports should record
the linked `ggml_version()` / `ggml_commit()`, the variable's presence, and
whether another component may have initialised Vulkan first.

Interleaved queue-off/queue-on runs used the documented baseline text, fixed
seed, current F16/Q8_0 models, and the fully optimised Kokoro graph. Ten F16
pairs and eight Q8_0 pairs gave:

| Model | Default queue median | Graphics-capable queue median | Paired median delta |
|---|---:|---:|---:|
| F16 | `65.16 ms` | `59.33 ms` | **`-6.02 ms` (-9.2%)** |
| Q8_0 | `76.24 ms` | `70.41 ms` | **`-5.94 ms` (-7.8%)** |

The F16 stage medians show why this is a workload-level choice rather than a
universally faster queue:

| Stage | Default queue | Graphics-capable queue | Paired median delta |
|---|---:|---:|---:|
| BERT + predictor | `19.81 ms` | `14.63 ms` | `-5.41 ms` |
| F0/noise | `10.82 ms` | `8.20 ms` | `-2.58 ms` |
| Text encoder | `3.74 ms` | `2.92 ms` | `-0.87 ms` |
| Decoder | `7.92 ms` | `8.88 ms` | `+0.77 ms` |
| Generator | `21.55 ms` | `23.52 ms` | `+1.68 ms` |
| Complete pipeline | `65.16 ms` | `59.33 ms` | **`-6.02 ms`** |

All ten F16 queue-off/on WAV pairs and all eight Q8_0 pairs were bit-identical.
The quick backend validation also passed with the override:

```bash
GGML_VK_ALLOW_GRAPHICS_QUEUE=1 \
  tools/run_backend_validation.sh \
    --backend vulkan --quick --skip-build
```

This is therefore a validated opt-in for the tested RADV/RX 7900 XTX system,
not a portable default. A graphics-capable queue may contend with a desktop,
game, compositor, or another Vulkan client, and other drivers/devices may make
different queue-family trade-offs. Hosts should A/B it on their own workload.

### Vulkan command-submit sizing — measured, not retained

The launch-heavy LSTM graphs made `GGML_VK_MAX_NODES_PER_SUBMIT` look like a
promising way to reduce command submission/fence overhead. GGML defaults to
100 graph nodes per submit. Six-run F16 sweeps found:

| Nodes per submit | Pipeline median |
|---:|---:|
| 25 | `65.84 ms` |
| 50 | `66.61 ms` |
| **100 (default)** | **`65.62 ms`** |
| 200 | `66.20 ms` |
| 400 | `67.96 ms` |
| 800 | `69.45 ms` |
| 1,600 | `69.15 ms` |
| 10,000 | `69.66 ms` |

The default was effectively optimal; increasing the command batch made the
pipeline slower. Do not assume that fewer Vulkan submissions improve this
graph without repeating the sweep after a material GGML/driver/workload change.

Forcing `GGML_VK_ASYNC_USE_TRANSFER_QUEUE=1` while using the graphics-capable
queue produced only a noisy approximately `-0.4 ms` paired median in eight
pairs, with no consistent stage-level signal. It is not retained as a
recommendation.

### Measured and rejected: pack generator style-conditioned affine projections

Historical selector `KOKORO_STYLE_PACK` (removed): pack the Vulkan generator's
48 AdaIN style matvecs into one `18432×128` matvec. CPU/non-Vulkan unchanged;
no GGML patch. Bit-identical (`sha256 a0457769aa8e…`); quick validation passed.

`GGML_VK_PERF_LOGGER=1`:

| Generator style matvec | Baseline (`KOKORO_STYLE_PACK=0`) | Packed (default) |
|---|---:|---:|
| `MUL_MAT(_ADD)_VEC` `m=512,k=128` (C=256) | 24 | 0 |
| `MUL_MAT(_ADD)_VEC` `m=256,k=128` (C=128) | 24 | 0 |
| `MUL_MAT_ADD_VEC` `m=18432,k=128` | 0 | 1 |
| Total generator style matvecs | **48** | **1** |

Twelve interleaved F16 pairs (`KOKORO_BENCH=1`, same fixture):

| Metric | Baseline (`KOKORO_STYLE_PACK=0`) | Packed (default) |
|---|---:|---:|
| Pipeline median | `60.68 ms` | `60.61 ms` |
| Generator median | `24.87 ms` | `24.83 ms` |
| Paired pipeline median delta | — | **`-0.74 ms`** |
| Paired generator median delta | — | **`-0.05 ms`** |
| Opt-faster pairs (pipeline) | — | `7/12` |
| Paired pipeline delta range | — | `-5.95 … +4.43 ms` |

**Not retained** — dispatch fell, wall time flat/noisy vs packing complexity.

### Measured and rejected: disable production diagnostic roots

Historical selector `KOKORO_DIAG_ROOTS` (removed): gate unused diagnostic graph
roots / forced CONT copies (`pred_lstm_out`, `pred_shared_out`, six
`pred_{f0,n}_{*}_out`, `dec_encode_out`, `gen_pre_post_out`). Production needs
only `dur_enc_out`, `durations_pre`, `f0_curve`, `n_curve`, `text_enc_out`,
`dec_decode_3_out`, `mag`, and `phase`. Bit-identical; model-free ctest and
quick validation passed.

Twelve interleaved F16 pairs (`KOKORO_BENCH=1`, same fixture):

| Metric | Diagnostic roots on (`KOKORO_DIAG_ROOTS=1`) | Roots gated off (default) |
|---|---:|---:|
| Pipeline median | `58.72 ms` | `58.75 ms` |
| Paired median delta | — | **`+0.10 ms`** |
| Opt-faster pairs | — | `6/12` |
| Paired delta range | — | `-1.00 … +3.91 ms` |

**Not retained** — flat/noisy vs code complexity.

### Measured and rejected: native ConvTranspose1d on the time-first generator

Historical selector `KOKORO_NATIVE_CONVT` (removed): replace the F32
`mul_mat + col2im_1d` upsample decomposition with native
`ggml_conv_transpose_1d` (F32 or F16). GGML Vulkan only ships an F32
`CONV_TRANSPOSE_1D` pipeline, so F16 mode CPU-offloads. Quick validation
passed; F16-model pairs near-identical but not bit-identical
(corr ≈ `0.99993`, rel_rms ≈ `0.0118`).

`GGML_VK_PERF_LOGGER=1`:

| Generator upsample path | Baseline | Native F32 (`KOKORO_NATIVE_CONVT=1`) |
|---|---:|---:|
| `COL2IM_1D` | 2 | 0 |
| `MUL_MAT f32` upsample (`m=5120` / `m=1536`) | 2 (~0.26 ms combined) | 0 |
| `CONV_TRANSPOSE_1D` | 0 | **2 (~6.53 ms + ~9.68 ms)** |

Twelve interleaved F16 pairs (`KOKORO_BENCH=1`, same fixture):

| Metric | Baseline (decomp) | Native F32 |
|---|---:|---:|
| Pipeline median | `59.07 ms` | `73.81 ms` |
| Generator median | `23.48 ms` | `37.75 ms` |
| Paired pipeline median delta | — | **`+13.95 ms`** |
| Paired generator median delta | — | **`+14.21 ms`** |
| Native-faster pairs (pipeline) | — | **`0/12`** |
| Paired pipeline delta range | — | `+11.76 … +19.34 ms` |

Native F16 smoke: pipeline `226.66 ms`, generator `191.59 ms` (CPU offload).
**Not retained** — materially slower than the existing decomposition.

### Measured and rejected: stage-selective `GGML_PREC_DEFAULT`

Historical selector `KOKORO_VK_MULMAT_DEFAULT` (removed): request
`GGML_PREC_DEFAULT` around selected graph builders instead of the usual
`GGML_PREC_F32` wrapper. Ordinary `ggml_conv_1d` already uses DEFAULT.

| Stage token | Wrapper `mul_mat` sites that flip to DEFAULT | PCM vs F32 wrapper |
|---|---|---|
| `decoder` | AdaIN style matvecs only (`n=1`) | **bit-identical** |
| `generator` | AdaIN style matvecs + 2 F32×F32 upsample decomp mats | **bit-identical** |
| `text` | 1 LSTM input proj (`n=L`) + recurrent matvecs | stable different hash |
| `f0` | shared LSTM / AdaIN projections | stable different hash |
| `bert_predictor` | BERT attentions/FFN + duration LSTMs/projections | stable different hash |

| Mode | Samples | vs F32 hash | corr / rel_rms vs F32 | Pipeline paired median (12) | Faster pairs |
|---|---:|---|---:|---:|---:|
| `text` | 61800 | changed | `0.999901` / `0.01404` | **+0.30 ms** | 5/12 |
| `f0` | 61800 | changed | `0.989255` / `0.1465` | −0.29 ms | 7/12 |
| `text,decoder,generator` | 61800 | same as `text` | `0.999901` / `0.01404` | **+0.48 ms** | 6/12 |
| `text,decoder,generator,f0` | 61800 | changed | `0.989329` / `0.1461` | +0.14 ms | 5/12 |
| `decoder` / `generator` alone | 61800 | **identical** | `1.0` / `0` | (no retention bench) | — |

Quick validation vs cached CPU:

| Mode | CPU/Vulkan corr | Relative RMS |
|---|---:|---:|
| unset F32 wrapper (baseline) | `0.970886` | `0.241052` |
| `text` | `0.971615` | `0.238003` |
| `text,decoder,generator` | `0.971615` | `0.238003` |
| `f0` | `0.886627` | `0.475426` |
| `bert_predictor` (diagnostic) | `0.936632` | `0.355386` |

**Not retained** — stages that change numerics are flat/noisy on latency;
decoder/generator DEFAULT is a no-op on this backend.

### Measured: graph lifecycle subphases; rejected per-stage schedulers

`KOKORO_BENCH=1` emits synchronized subphase timers for each GPU stage
(`bert_predictor`, `f0_noise`, `text_encoder`, `decoder`, `generator`) at
existing sync points only (no extra `ggml_backend_sched_synchronize`):

| Subphase | What it covers |
|---|---|
| `graph_build` | `kokoro_build_graph_*` |
| `reset_alloc` | `ggml_backend_sched_reset` + `alloc_graph` |
| `upload` | input `ggml_backend_tensor_set` |
| `compute` | `ggml_backend_sched_graph_compute` (includes its sync) |
| `download` | output `ggml_backend_tensor_get` |
| `host` | unavoidable host prep (lookups, vector alloc, frees) |

Fifteen process runs after two warmups (same fixture). WAV hash stable
(`a0457769aa8e…`). Subphase medians (ms):

| Stage | graph_build | reset_alloc | upload | compute | download | host | Stage total |
|---|---:|---:|---:|---:|---:|---:|---:|
| BERT + predictor | `0.31` | `1.58` | `0.00` | `12.90` | `0.15` | `0.05` | `15.02` |
| F0/noise | `0.20` | `1.06` | `0.01` | `6.87` | `0.11` | `0.03` | `8.30` |
| Text encoder | `0.08` | `0.50` | `0.00` | `2.21` | `0.07` | `0.01` | `2.87` |
| Decoder | `0.05` | `0.26` | `0.00` | `8.22` | `0.15` | `0.02` | `8.71` |
| Generator | `0.17` | `0.64` | `0.08` | `22.27` | `0.32` | `0.08` | `23.59` |
| **Sum** | **`0.81`** | **`4.04`** | **`0.09`** | **`52.47`** | **`0.80`** | **`0.19`** | — |

Pipeline median `59.93 ms`. Host-side lifecycle ceiling (everything except
`compute`) is **`5.93 ms` (~10%)**, dominated by `reset_alloc` (`4.04 ms`).
Transfers are tiny (`0.89 ms` combined).

Historical selector `KOKORO_STAGE_SCHED` (removed): five independent schedulers
to keep per-stage gallocr plans warm. Bit-identical; quick validation matched
baseline. CLI gained `KOKORO_BENCH_ITERS` (requires `KOKORO_BENCH=1`) for the
warm measurement — that CLI hook is retained.

Twelve interleaved cold process pairs:

| Metric | Baseline | `KOKORO_STAGE_SCHED=1` |
|---|---:|---:|
| Pipeline median | `59.92 ms` | `61.55 ms` |
| Paired pipeline median delta | — | **`+1.55 ms`** |
| Opt-faster pairs | — | **`1/12`** |
| Paired delta range | — | `-0.48 … +6.28 ms` |

Twelve interleaved in-process warm pairs (`KOKORO_BENCH_ITERS=6`, iters 1…5):

| Metric | Baseline warm | Stage-sched warm |
|---|---:|---:|
| Pipeline median (all warms) | `44.23 ms` | `43.44 ms` |
| Paired warm pipeline median delta | — | **`-0.35 ms`** |
| Opt-faster pairs | — | `8/12` |
| Paired delta range | — | `-3.71 … +2.42 ms` |
| Warm `reset_alloc` sum median | `4.00 ms` | `3.63 ms` (`-0.41 ms` paired) |

**Not retained** — cold regresses; warm flat/noisy. Most of the `4.04 ms` is
structural `split_graph`, not buffer re-reserve. Subphase timers kept.
Persistent-transfer / graph-cache follow-ups are deprioritized by these ceilings.

### Measured: shared predictor reverse-time index; rejected

Historical selector `KOKORO_LSTM_SHARE_REV_IDX` (removed): share one functional
I32 reverse index across the four same-length predictor LSTMs. Bit-identical;
quick validation matched baseline.

`GGML_VK_PERF_LOGGER=1` on `bert_predictor` (`CAST` logged as `CPY`):

| Op | Baseline (`SHARE_REV_IDX=0`) | Shared (default) | Delta |
|---|---:|---:|---:|
| `ARANGE` | 8 | 2 | −6 |
| `SCALE` | 4 | 1 | −3 |
| `ADD` | 320 | 317 | −3 |
| `CPY` (cast) | 5 | 2 | −3 |
| `GET_ROWS` | 10 | 10 | 0 |

Twelve interleaved process-per-run F16 pairs (same fixture):

| Metric | Baseline (`KOKORO_LSTM_SHARE_REV_IDX=0`) | Shared (default) |
|---|---:|---:|
| Pipeline median | `59.23 ms` | `59.94 ms` |
| `bert_predictor` median | `14.61 ms` | `14.87 ms` |
| Paired pipeline median delta | — | **`+0.61 ms`** |
| Paired `bert_predictor` median delta | — | **`+0.26 ms`** |
| Opt-faster pairs (pipeline) | — | **`4/12`** |
| Paired pipeline delta range | — | `-4.58 … +3.92 ms` |

**Not retained** — dispatch saving is real but below wall-time noise.

### Measured: persistent LSTM gate reorder `i,f,g,o` → `i,f,o,g`; rejected

Historical selector `KOKORO_LSTM_GATE_REORDER` (removed, F16-only): persistently
swap `g`/`o` gate rows so sigmoid can be one contiguous `3H` view. Bit-identical;
quick validation matched baseline. Q8_0 not extended.

`GGML_VK_PERF_LOGGER=1`:

| Op launches (sum of stage blocks) | Baseline (`GATE_REORDER=0`) | Reordered (default) | Delta |
|---|---:|---:|---:|
| `SIGMOID` | **979** (`537+308+134`) | **329** (`181+103+45`) | **−650** |
| `TANH` | 656 | 656 | 0 |

Twelve interleaved process-per-run F16 pairs (`KOKORO_BENCH=1`, same fixture):

| Metric | Baseline (`KOKORO_LSTM_GATE_REORDER=0`) | Reordered (default) |
|---|---:|---:|
| Pipeline median | `60.20 ms` | `60.56 ms` |
| Paired pipeline median delta | — | **`+1.01 ms`** |
| Opt-faster pairs (pipeline) | — | **`4/12`** |
| Paired pipeline delta range | — | `-2.05 … +3.87 ms` |

**Not retained** — wider `3H` sigmoid offsets the dispatch win (same lesson as
contiguous I/F batching).

### Remaining LSTM micro-optimisations

No existing GGML operation provides an exact LSTM scan. Application-only ideas
on this fixture are low-ceiling:

- Done / rejected: shared reverse-time index and gate reorder (above).
- Deprioritized: persistently prepack direction weights/biases.

Vulkan can fuse `MUL_MAT → ADD` / `MUL_MAT → ADD → ADD` for single-row matvecs
with matching bias shape; direction-batched LSTM uses `ne[2]=2` and is not
eligible. Do not disable direction batching to recover that fusion.

### Measured and rejected: existing-op device harmonic (`KOKORO_DEVICE_HAR`)

**Formulation (existing GGML ops, no kernel patch):**

```text
nearest interpolate F0
harmonic multipliers and fractional part
linear downsample
cumsum × 2π
linear upsample
sin
voicing/noise affine
Linear(9→1) + tanh
fixed STFT matrix
```

using `ggml_interpolate`, `ggml_floor`, `ggml_cumsum`, `ggml_sin`, `ggml_tanh`,
matmul, and reflection padding. Host still supplies fixed-seed `rand_ini` and
Gaussian noise. Prototype kept STFT on host (no `atan2`; large `PAD_REFLECT_1D`
is CPU-fallback).

Ceiling first (no graph change), same fixture, n=8:

| Fixture | Samples | `harmonic_source` med | `harmonic_wait` med | text+decoder med | `pipeline_total` med |
|---|---:|---:|---:|---:|---:|
| short (`Hello.`) | 31,800 | 5.13 ms | **0.01 ms** | 11.04 ms | 41.35 ms |
| baseline | 61,800 | 9.91 ms | **0.01 ms** | 11.38 ms | 58.52 ms |
| punctuation | 127,200 | 20.55 ms | **9.91 ms** | 10.69 ms | 114.61 ms |
| long | 313,200 | 49.26 ms | **32.80 ms** | 16.44 ms | 217.11 ms |

Baseline-only “wait≈0” does **not** hold: punctuation/long leave ~10–33 ms of
CPU harmonic on the critical path. Retained `har_phase.*` timers on long:
`upsample_sin` ~14 ms, `noise_msource` ~20 ms, `stft` ~9 ms.

Historical selector `KOKORO_DEVICE_HAR` (removed): host RNG/noise overlap +
existing-op `har_source` graph + host STFT. Noise H2D ~9·L_high vs current har
upload ~22·T_har; same compute queue as the rest of the pipeline. CPU-backend
graph vs host `har_source` matched (`corr=1`). Quick validation with the
selector passed (`corr=0.969428`).

| Fixture | Pipeline paired median Δ | Device-faster | Paired range | Vulkan↔Vulkan corr med |
|---|---:|---:|---|---:|
| baseline | **+9.48 ms** | **0/12** | +6.76…+11.84 | 0.99957 |
| punctuation | **+2.66 ms** | 1/12 | −0.35…+5.30 | 0.99966 |
| long | **−2.24 ms** | 9/12 | −4.73…+3.40 | 0.99856 |

**Not retained** — baseline regresses; long win noisy/~2 ms. Revisit only with a
second queue, fused device RNG+STFT (`atan2`), or long/batched throughput.
`har_phase.*` collection stays (worker collects; caller logs after join).

### Measured and rejected: Vulkan device iSTFT via existing GGML ops

**Formulation (existing GGML ops, no kernel patch):**

```text
real = mag * cos(phase);  imag = mag * sin(phase)
# DC real forced to mag (phase ignored) to match historical host iSTFT
frames = IDFT_20x22 @ concat(real, imag)   # fixed Hermitian real inverse DFT
frames *= periodic_Hann_20
wave = col2im_1d(frames, stride=5, oc=1)   # overlap-add
norm = col2im_1d(repeat(Hann²), stride=5, oc=1)
pcm  = center_crop(wave / norm)            # length (T_har-1)*5
```

Historical selector `KOKORO_DEVICE_ISTFT` (removed): append the above to the
Vulkan generator and download PCM. Focused CPU-backend parity matched host
iSTFT (max abs ≤ `4.5e-8`, corr `1.0`). Quick validation passed for baseline
and device (`corr=0.970887`). Twelve interleaved pairs: corr `0.99999924`,
rel_rms `1.253e-3`.

`GGML_VK_PERF_LOGGER=1` (generator stage block):

| Op (generator timings block) | Baseline (CPU iSTFT) | Device iSTFT |
|---|---:|---:|
| `COL2IM_1D` (upsample) | 2 | 2 |
| `COL2IM_1D` (iSTFT wave+norm) | 0 | **2 (~8 µs)** |
| `MUL_MAT f32` `m=20,n=T_har,k=22` (IDFT) | 0 | **1 (~10 µs)** |
| `COS` / extra `SIN` | 0 / 1 | **1 / 2** |
| Generator download median | `0.35 ms` (mag+phase) | **`0.12 ms` (pcm)** |
| Host `istft` median | `1.08 ms` | `0` (in-graph) |

Twelve interleaved F16 pairs (`KOKORO_BENCH=1`, same fixture):

| Metric | Baseline (CPU iSTFT) | Device iSTFT |
|---|---:|---:|
| Pipeline median | `59.12 ms` | `58.42 ms` |
| Generator median | `23.62 ms` | `23.91 ms` |
| `generator.compute` median | `22.28 ms` | `22.69 ms` |
| Paired pipeline median delta | — | **`−0.87 ms`** |
| Paired gen+iSTFT median delta | — | **`−0.57 ms`** |
| Device-faster pairs (pipeline) | — | **`7/12`** |
| Paired pipeline delta range | — | `−5.65 … +2.94 ms` |

**Not retained** — ~1 ms Amdahl ceiling does not survive as stable wall-time;
generator compute/alloc absorbs most of the host-iSTFT + download savings.
Revisit only for long/batched device-resident pipelines.

### Cumulative audit decision (final review)

This follow-up no-kernel-patch audit keeps **one** host-configured latency
optimisation and bench-only instrumentation. All measured application-graph
and GGML-tuning experiments were reverted.

**Retained (latency)**

- Host opt-in `GGML_VK_ALLOW_GRAPHICS_QUEUE=1` before Vulkan device init
  (RADV/RX 7900 XTX: F16 paired median about −6 ms / −9%; Q8_0 about −6 ms /
  −8%; bit-identical WAVs). Not a portable library default.

**Retained (instrumentation only; no normal-path behaviour change when unset)**

- `KOKORO_BENCH=1` stage timers, GPU lifecycle subphases
  (`*.graph_build` / `*.reset_alloc` / `*.upload` / `*.compute` / `*.download` /
  `*.host`), and CPU `har_phase.*` breakdown.
- Harmonic subphase times are collected on the worker and reported only after
  join on the caller thread (same policy as `harmonic_source` / `work_ms`). The
  Kokoro log callback is never invoked from the harmonic worker.
- CLI `KOKORO_BENCH_ITERS=N` in-process repeats, **only when** `KOKORO_BENCH=1`
  is also set (avoids surprising multi-synthesis CLI behaviour).

**Rejected / reverted (measured in this audit)**

- `GGML_VK_MAX_NODES_PER_SUBMIT` sweeps (default 100 best); transfer-queue force
- Generator AdaIN style packing (`KOKORO_STYLE_PACK`)
- Gating unused diagnostic roots (`KOKORO_DIAG_ROOTS`)
- Native time-first `ggml_conv_transpose_1d` (`KOKORO_NATIVE_CONVT`)
- Stage-selective `GGML_PREC_DEFAULT` (`KOKORO_VK_MULMAT_DEFAULT`)
- Per-stage schedulers (`KOKORO_STAGE_SCHED`)
- Shared predictor reverse-time index (`KOKORO_LSTM_SHARE_REV_IDX`)
- Persistent LSTM gate reorder (`KOKORO_LSTM_GATE_REORDER`)
- Existing-op device harmonic (`KOKORO_DEVICE_HAR`)
- Vulkan device iSTFT via existing ops (`KOKORO_DEVICE_ISTFT`)

No GGML patches. No revived selector/prepack/device-graph/test sources remain
in the tree.

**Full default validation (final review)**

```bash
GGML_VK_ALLOW_GRAPHICS_QUEUE=1 \
  tools/run_backend_validation.sh --backend vulkan
```

Outcome after final review cleanup (RX 7900 XTX, fixed seed, baseline fixture,
2 process runs/model):

| Model | Samples | CPU↔Vulkan corr | Rel RMS | Deterministic Vulkan WAV | Result |
|---|---:|---:|---:|---|---|
| F16 | 61800 | 0.953395 | 0.304968 | yes (`a0457769aa8e…`) | **pass** |
| Q8_0 | 61800 | 0.690435 | 0.785729 | yes (`8cb64dbfedd4…`) | **pass** |

Model-free `ctest` in `.validation/build-vulkan` also passed (5/5). Known
full-corpus long-utterance CPU/Vulkan duration/sample-count mismatch is
unchanged and was **not** weakened; default (non-`--full-corpus`) validation
does not exercise that gate.

## Explored approaches that were not retained

Rejected experiments are important results. Revisit them only when a backend,
kernel, graph allocator, or workload assumption has changed.

| Experiment | Measured result | Decision / learning |
|---|---:|---|
| AdaIN affine as `(1+γ)·x+β` | Pipeline −0.33 ms | Removed many full-tensor ADDs in profiling but did not move wall time enough to justify changed FP ordering. |
| Merge TextEncoder → Decoder → Generator | Pipeline +0.41 ms | Transfers were small; scheduling/allocation of the larger graph offset two removed submissions. |
| Batch contiguous LSTM input/forget sigmoid | Pipeline −0.22 ms | Dispatch count fell, but each doubled-width sigmoid took longer. |
| Time-first F0/noise + decoder residual stacks | Pipeline −0.27 ms | CONT count fell from 184 to 79, but those copies represented only about 0.8 ms GPU time. |
| Batch F0 and noise branches | Pipeline median −1.35 ms; paired median −0.82 ms | Correct and potentially useful on a higher-launch-cost backend, but too much code for the measured Vulkan gain. Native `conv_1d` could not bind per-batch weights, requiring im2col + batched matmul. |
| Batch generator K3/K7/K11 residual branches | Pipeline **+1.15 ms** | Slower. Padding all kernels to K11 increased relevant convolution FLOPs by about 57%; the generator is compute-bound enough that fewer launches did not compensate. |
| Replace copy boundaries by one very large post-duration graph | Flat/slightly slower | Device transfers are no longer the dominant cost at these compact boundaries. |
| Gate unused diagnostic graph roots / forced CONT copies | Pipeline median +0.03 ms; paired median **+0.10 ms**; 6/12 faster; range −1.00…+3.91 ms | Bit-identical and low-risk, but flat/noisy vs code complexity on RADV with graphics queue. Reverted. |
| Pack generator AdaIN style projections (48→1 matvec) | Pipeline median −0.07 ms; paired median **−0.74 ms**; generator paired **−0.05 ms**; 7/12 faster; range −5.95…+4.43 ms | Bit-identical; PERF_LOGGER showed 48→1 generator style matvecs, but wall time flat/noisy vs packing complexity. Reverted. |
| Native time-first `ggml_conv_transpose_1d` vs F32 mul_mat+col2im | Pipeline median **+14.74 ms**; paired median **+13.95 ms**; generator paired **+14.21 ms**; **0/12** faster; range +11.76…+19.34 ms | Correct enough (corr≈0.99993) but much slower; PERF_LOGGER showed 2×`CONV_TRANSPOSE_1D` (~16.2 ms) replacing ~0.37 ms decomp. Native F16 CPU-offloads (gen ~192 ms). Reverted. |
| Stage-selective `GGML_PREC_DEFAULT` (`KOKORO_VK_MULMAT_DEFAULT`) | `text` paired **+0.30 ms** (5/12); `f0` paired −0.29 ms (7/12); `text+decoder+generator` paired **+0.48 ms** | Decoder/generator bit-identical (matvecs + F32 decomp; `conv_1d` already DEFAULT). Text/f0 change numerics without stable latency. BERT/predictor diagnostic worsens PCM; not retained. Reverted. |
| Per-stage schedulers (`KOKORO_STAGE_SCHED`) | Cold paired **+1.55 ms** (1/12); warm in-process paired **−0.35 ms** (8/12); warm `reset_alloc` sum only **−0.41 ms** | Host ceiling ~`5.9 ms` (`reset_alloc` 4.0). Most cost is structural `split_graph`, not re-reserve. Bit-identical; not retained. Subphase timers kept. |
| Share predictor reverse-time I32 index (`KOKORO_LSTM_SHARE_REV_IDX`) | Pipeline paired **+0.61 ms** (4/12); `bert_predictor` paired **+0.26 ms**; range −4.58…+3.92 ms | Bit-identical; PERF_LOGGER `ARANGE` 8→2 / `SCALE` 4→1 / `ADD`/`CPY` −3 in `bert_predictor`, but wall time flat/noisy vs helper/API complexity. Reverted. |
| Vulkan device iSTFT (`KOKORO_DEVICE_ISTFT`, existing GGML ops) | Pipeline paired **−0.87 ms** (7/12); gen+iSTFT paired **−0.57 ms**; range −5.65…+2.94 ms | Near-identical (corr `0.99999924`); PERF_LOGGER IDFT/col2im ~18 µs but generator.compute +0.41 ms absorbs host 1.08 ms + download 0.23 ms. Flat/noisy vs complexity. Reverted. |
| Persistent LSTM gate reorder `i,f,g,o`→`i,f,o,g` (`KOKORO_LSTM_GATE_REORDER`, F16) | Pipeline paired **+1.01 ms** (4/12); range −2.05…+3.87 ms; stage paired mostly ≤0.3 ms except generator `+0.72` | Bit-identical; PERF_LOGGER `SIGMOID` **979→329**, `TANH` unchanged 656. Wider 3H sigmoid offsets dispatch savings (same lesson as I/F batching). Reverted; Q8_0 not extended. |
| Existing-op device harmonic (`KOKORO_DEVICE_HAR`, host RNG + GPU `har_source` + host STFT) | Baseline paired **+9.48 ms** (0/12); punct **+2.66 ms** (1/12); long **−2.24 ms** (9/12); range long −4.73…+3.40 | Ceiling wait is real on punct/long (~10–33 ms) but same-queue serial device path + larger noise H2D loses short/baseline overlap; long win noisy/~2 ms. CPU graph parity fine; Vulkan PCM corr ~0.999. Reverted. |

Other low-ceiling ideas remain possible but are not current priorities:

- persistently prepacking batched LSTM weights removes only a small subset of
  the remaining concat work;
- precomputing Snake reciprocals removes less than 1 ms of measured setup;
- moving the ~1 ms iSTFT to a graph was measured (fixed DFT + `col2im_1d` OLA):
  numerically fine but flat/noisy on this fixture; not retained;
- moving harmonic generation to the GPU was measured across short/baseline/
  punctuation/long: wait is material on longer fixtures, but the least-complex
  existing-op split does not beat overlapped CPU har on this single-queue RADV
  path; not retained.

Meaningful next steps probably require one of:

1. a native/fused recurrent operation that performs an LSTM timestep (or
   sequence scan) without thousands of elementwise launches;
2. safe concurrent GPU queues for independent F0/noise and text-encoder work;
3. multi-utterance batching for throughput rather than single-utterance
   latency;
4. a fused/portable device-side harmonic/iSTFT region only when longer or
   batched workloads make CPU overlap insufficient — existing-op prototypes
   for both were measured and rejected for single-utterance RADV latency.

The follow-up audit above adds useful no-kernel work before those larger steps,
but—with the exception of RADV queue selection—its estimated gains are mostly
in the sub-2-ms range. It does not change the conclusion that a materially
larger recurrent improvement requires backend/kernel support or a different
throughput-oriented scheduling objective.

## Long-text duration parity finding

The extended corpus exposed a correctness boundary that the short baseline did
not:

| Model | CPU samples | Vulkan samples | Difference |
|---|---:|---:|---:|
| F16 long fixture | 313,800 | 313,200 | −1 duration frame |
| Q8_0 long fixture | 312,000 | 313,200 | +2 duration frames |

`KOKORO_DEBUG_DURATIONS=1` logs every raw/scaled/rounded token duration through
the Kokoro callback. This localized the first discrete disagreement to
`durations_pre` values on opposite sides of half-frame rounding boundaries.
Examples:

| Model/token | CPU | Vulkan | Rounded CPU/GPU |
|---|---:|---:|---:|
| F16 / 55 | 3.497143 | 3.505643 | 3 / 4 |
| F16 / 80 | 19.502647 | 19.499670 | 20 / 19 |
| F16 / 195 | 2.503793 | 2.499181 | 3 / 2 |
| Q8_0 / 73 | 1.498011 | 1.507703 | 1 / 2 |
| Q8_0 / 195 | 2.479589 | 2.515164 | 2 / 3 |

The batched-LSTM optimisation is not the cause: disabling
`KOKORO_LSTM_BATCH` produced the same long-fixture sample mismatch.

An attempted 1/16-frame quantisation before integer rounding was rejected. It
made the originally observed boundary values agree, but introduced new
quantisation boundaries: two F16 values that previously both rounded to one
frame became one versus two, increasing the sample-count mismatch. Arbitrary
rounding grids or fixture-specific epsilon shifts are not a correctness fix.

The full-corpus harness therefore continues to fail the strict equal-sample
gate for this fixture. Do not hide this by weakening the gate. Proper follow-up
is stage-level comparison through BERT, duration encoder, predictor LSTM, and
duration projection, followed by either improved backend numerical parity or a
deliberate model-level duration policy with corpus-wide evidence.

### Phase B — stable intermediate-stage diagnostics

Add a test-only observer or diagnostic executable that captures existing host
boundaries without changing graph scheduling. Required stages:

- `bert_proj`;
- `dur_enc` and raw durations;
- F0 and noise curves;
- `text_enc`;
- decoder output;
- generator magnitude and phase;
- PCM.

The observer must be disabled by default, avoid process-global state, and
clearly define tensor shape/layout. Stage comparison should reuse the same
metric implementation as WAV comparison.

### Phase C — make current graphs backend-safe

- Audit all copy-to-view mutation and in-place operations.
- Add focused synthetic tests for recurrent output assembly and aliased views.
- Establish F16 per-stage tolerances on Vulkan, then Metal and CUDA.
- Investigate Q8_0 Vulkan kernel drift independently from phase reconstruction.
- Verify which nodes fall back to CPU.

### Phase D — remove avoidable host round-trips

Completed on the current tree:

- Keep BERT output device-resident into the predictor where shape permits
  (combined BERT + duration-predictor graph; only `dur_enc_out` and
  `durations_pre` cross to the host before the dynamic-duration barrier).
- Build alignment index tensors on the host and execute alignment as device
  gathers for duration and text encodings (`ggml_get_rows`).
- Overlap CPU harmonic generation with TextEncoder + Decoder on Vulkan after
  hosting `m_source` weights; join before the generator.

Measured and **deprioritized** on this RADV single-utterance fixture (see
follow-up audit / explored-approaches table; not immediate open work):

- Persistent boundary tensors / keeping predictor–text–F0 encodings resident
  across the five-stage shared scheduler: transfer slice is only ~0.9 ms;
  host lifecycle is dominated by structural `split_graph` in `reset_alloc`.
- Connecting decoder output directly to the generator / one large
  post-duration graph: previously measured flat/slightly slower; transfers
  are not the dominant cost at these compact boundaries.
- Per-stage schedulers / reserve-driven reuse: cold path regressed; warm path
  flat/noisy with only a partial `reset_alloc` reduction.

### Phase E — enlarge the native compute region

Measured and **rejected / deprioritized** on this fixture (not immediate open
work; revisit only under the stated conditions):

- Existing-op device harmonic: real unhidden wait on punctuation/long, but
  same-queue device path regresses baseline and only yields a noisy ~2 ms
  long-fixture paired median. Revisit with a second queue, fused device
  RNG/STFT (`atan2`), or long/batched throughput.
- Device iSTFT via existing GGML ops: near-parity, flat/noisy ~1 ms ceiling.
  Revisit only with a fused kernel or long/batched device-resident need.
- Per-stage scheduler reuse / larger graph-cache work: not justified by the
  ~6 ms host lifecycle ceiling measured above.

Still dependent on backend/kernel capability rather than application-only
rewrites:

- a native/fused recurrent LSTM timestep or sequence scan;
- safe concurrent GPU queues for independent F0/noise and text-encoder work;
- multi-utterance batching for throughput rather than single-utterance latency.

### Retained: Highway SIMD harmonic source

Fresh profiling after the earlier Vulkan work showed that the host harmonic
source remained on the critical path for longer utterances:

| F16 fixture | Pipeline | Harmonic worker | Unhidden harmonic wait |
|---|---:|---:|---:|
| baseline text | `39.74 ms` | `9.45 ms` | `2.56 ms` |
| long fixture | `172.68 ms` | `46.73 ms` | `33.69 ms` |

The scalar hotspots were phase interpolation plus `sinf`, Gaussian noise from
`std::normal_distribution`, and the direct 20-point STFT (`sqrtf`/`atan2f`).
The retained implementation uses the permissively licensed Google Highway
library with runtime ISA dispatch:

- vector sine for the upsampled harmonic phase;
- vector Xoshiro + Box-Muller Gaussian noise;
- an 11-bin SIMD DFT with vector magnitude and phase.

Eight interleaved process pairs used `KOKORO_BENCH_ITERS=4`; iteration zero was
discarded and each process contributed the median of its three warm runs.
Every optimized pair was faster:

| F16 fixture/stage | Baseline | Highway | Paired median delta |
|---|---:|---:|---:|
| baseline harmonic source | `9.45 ms` | `2.33 ms` | `-7.07 ms` |
| baseline harmonic wait | `2.56 ms` | `0.00 ms` | `-2.56 ms` |
| baseline pipeline | `39.74 ms` | `37.12 ms` | **`-2.41 ms`** |
| long harmonic source | `46.73 ms` | `14.38 ms` | `-32.39 ms` |
| long harmonic wait | `33.69 ms` | `1.31 ms` | `-32.47 ms` |
| long pipeline | `172.68 ms` | `132.68 ms` | **`-40.95 ms`** |

The short harmonic worker is `4.06x` faster and becomes fully hidden by text
encoder + decoder work. The long worker is `3.25x` faster and nearly hidden.
The long generator also improved by a paired `8.21 ms` under the shorter CPU
wait; as with the earlier overlap result, this is an observed scheduling/device
clock effect rather than changed generator graph math.

The fast Gaussian stream deliberately changes fixed-seed PCM. Against the
historical seed-1234 output, long-fixture correlation is `0.99313` and relative
RMS delta is `0.11714`. This matches ordinary historical seed variation
(`0.99327`, `0.11599` for seed 1235), and repeated fast seed-1234 runs are
bit-identical. `KOKORO_LEGACY_RNG=1` retains the historical
`std::normal_distribution` stream for comparisons; with legacy noise, the SIMD
sine/STFT-only output has correlation `0.99993`.

## Definition of “GPU support works properly”

A backend is considered supported when:

- build and explicit runtime selection work;
- unsupported devices fail clearly;
- fixed-seed output is deterministic;
- hard output invariants hold across the validation corpus;
- transcripts match the CPU reference;
- no unexplained early-stage numerical discontinuity remains;
- backend-specific numerical baselines and known limitations are documented;
- the repeatable validation command passes on representative hardware.

Performance improvement is desirable, but it is not part of the correctness
definition.
