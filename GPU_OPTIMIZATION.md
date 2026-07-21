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
duration predictor:

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

Representative F16 pipeline medians on the RX 7900 XTX evolved as follows:

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

At this point the largest remaining regions are approximately:

- BERT + duration predictor: 20 ms;
- generator: 21 ms;
- F0/noise: 10–11 ms;
- decoder: 7–8 ms;
- text encoder: 3–4 ms.

The generator's large convolutions already reach high throughput in Vulkan
`mul_mat` kernels. The recurrent regions remain launch-heavy even after
direction batching: their timestep dependency prevents batching across time,
and there is no native GGML LSTM operation.

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

Other low-ceiling ideas remain possible but are not current priorities:

- persistently prepacking batched LSTM weights removes only a small subset of
  the remaining concat work;
- precomputing Snake reciprocals removes less than 1 ms of measured setup;
- moving the ~1 ms iSTFT to a graph is possible with fixed DFT matrices and
  overlap-add, but is complex relative to its Amdahl ceiling;
- moving harmonic generation to the GPU does not improve the current fixture
  while its CPU work is fully hidden.

Meaningful next steps probably require one of:

1. a native/fused recurrent operation that performs an LSTM timestep (or
   sequence scan) without thousands of elementwise launches;
2. safe concurrent GPU queues for independent F0/noise and text-encoder work;
3. multi-utterance batching for throughput rather than single-utterance
   latency;
4. a portable device-side harmonic/iSTFT region when CPU overlap is
   insufficient for longer workloads.

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

- Done: keep BERT output device-resident into the predictor where shape
  permits (combined BERT + duration-predictor graph; only `dur_enc_out` and
  `durations_pre` cross to the host before the dynamic-duration barrier).
- Read only the small duration vector.
- Done: build alignment index tensors on the host and execute alignment as
  device gathers for duration and text encodings (`ggml_get_rows`).
- Keep predictor/text-encoder outputs device-resident across scheduler resets
  so compact encodings are not downloaded only to be re-uploaded.
- Keep F0/noise tensors resident for the decoder while downloading only the F0
  data required by the current CPU harmonic generator.
- Done (Vulkan): overlap CPU harmonic generation with TextEncoder + Decoder
  after hosting `m_source` weights; join before the generator.
- Connect decoder output directly to the generator.

Each merge should remove one boundary and retain the previous path as a
reference until parity is established.

### Phase E — enlarge the native compute region

- Implement harmonic generation with portable GGML operations or a justified
  native backend operation.
- Implement and validate device iSTFT.
- Merge compatible stage graphs and reuse scheduler allocations.
- Reduce submissions and synchronisation only after stage-level correctness is
  stable.

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
