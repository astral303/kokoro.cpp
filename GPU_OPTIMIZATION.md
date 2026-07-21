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

The current implementation intentionally uses CPU-friendly stage boundaries:

```text
1.  BERT graph
      download bert_proj
2.  Duration predictor graph
      download duration encoding and duration values
3.  CPU duration rounding and alignment
      upload aligned duration encoding
4.  F0/noise graph
      download F0 and noise curves
5.  Text encoder graph
      download text encoding
6.  CPU text alignment
      upload aligned text encoding
7.  Decoder graph
      download decoder activation
8.  CPU harmonic source generation
      upload decoder activation and harmonics
9.  Generator graph
      download magnitude and phase
10. CPU iSTFT
      return PCM
```

This was reasonable for the original CPU-only implementation: a backend
upload/download was generally a host-memory copy, and independent stages were
easier to validate. On a discrete GPU, the same structure introduces repeated
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

It does not require downloading the full duration encoding or text encoding.
After the duration indices are known, alignment can become a device gather
using a small host-generated index tensor.

### 5. CPU custom stages force additional boundaries

The following stages currently use ordinary C++ rather than GGML graphs:

- duration/text alignment;
- harmonic source and random-noise generation;
- final iSTFT.

The harmonic source upload and magnitude/phase download are therefore real
boundaries today. Moving these stages must be done only after native-kernel
parity tests exist.

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

- Keep BERT output device-resident into the predictor where shape permits.
- Read only the small duration vector.
- Build alignment index tensors on the host and execute alignment as device
  gathers for duration and text encodings.
- Keep F0/noise tensors resident for the decoder while downloading only the F0
  data required by the current CPU harmonic generator.
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
