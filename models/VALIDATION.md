# Official conversion validation

This report records the July 21, 2026 comparison between GGUF files published
by `cstr` and files generated directly from pinned
`hexgrad/Kokoro-82M` revision
`f3ff3571791e39611d31c381e3a41a3af07b4987`.

## Sizes

| Artifact | cstr | Generated here | Difference |
|---|---:|---:|---:|
| F16 model | 163,728,096 | 163,728,096 | 0 bytes |
| Q8_0 model | 141,322,336 | 141,322,752 | +416 bytes |
| Shared voice pack | 522,560 | 522,560 | 0 bytes |
| All 54 generated voices | n/a | 28,218,272 | n/a |

The 416-byte Q8_0 difference is container metadata/alignment; both files have
459 tensors with the same names and the same type counts: 111 Q8_0, 142 F16,
and 206 F32.

Most generated voices are 522,560 bytes; `jf_gongitsune` is 522,592 bytes
because its longer metadata name changes GGUF alignment. The cstr voice
repository has three voices that also occur in the official
Hexgrad v1.0 voice set: `af_heart`, `ef_dora`, and `ff_siwis`. All three
published `voice.pack` tensors are element-for-element identical to our F32
conversions (max error and RMSE both zero). Its four `df_*`/`dm_*` voices are
for the separate German model and have no counterpart in the official v1.0
voice set.

## Tensor errors

Every statistic below is computed after dequantization over all 81,731,256
model elements.

### cstr versus generated here

| Type | Max absolute | RMSE | MAE | Relative L2 | Cosine |
|---|---:|---:|---:|---:|---:|
| F16 | 2.89440155e-4 | 3.33034834e-7 | 3.16075518e-9 | 5.34848643e-6 | 0.999999999986 |
| Q8_0 | 3.76281738e-2 | 1.06066509e-4 | 8.01437548e-6 | 1.70338414e-3 | 0.999998549352 |

For F16, 376 of 459 tensors are exact and 83 differ. The largest difference is
in `pred.N.2.conv2.weight`. The difference is intentional: this repository
uses PyTorch's actual weight-normalization formula,
`v * (g / clamp_min(norm, eps))`. The inherited converter used
`v * clamp_min(g / norm, eps)`, which changes negative learned `weight_g`
values. The corrected formula was checked directly against
`torch._weight_norm` (maximum float32 discrepancy 1.49e-8).

Q8_0 files are independently quantized, so direct Q8-to-Q8 element differences
include different rounding decisions. Comparing each Q8 model with its own F16
baseline is more meaningful:

| Pair | Max absolute | RMSE | MAE | Relative L2 |
|---|---:|---:|---:|---:|
| cstr Q8_0 versus cstr F16 | 1.91650391e-2 | 3.19161067e-4 | 1.16402241e-4 | 5.12567592e-3 |
| Our Q8_0 versus our F16 | 1.91955566e-2 | 3.19376048e-4 | 1.16557019e-4 | 5.12912849e-3 |

The quantization-error distributions are therefore nearly identical; our
global RMSE is 0.067% higher in this comparison.

## Fixed-seed audio output

The four model files synthesized the same sentence with `af_heart`, CPU, four
threads, length scale 1.0, and `KOKORO_SEED=42`. All outputs had exactly 61,800
PCM16 samples (2.575 seconds at 24 kHz).

| Reference -> candidate | RMS delta | Relative RMS | Correlation | SNR dB | Spectral convergence | Log-spectral distance dB |
|---|---:|---:|---:|---:|---:|---:|
| cstr F16 -> our F16 | 6.2137e-4 | 0.01267 | 0.999920 | 37.94 | 0.00549 | 0.529 |
| cstr F16 -> cstr Q8_0 | 0.03647 | 0.74362 | 0.72357 | 2.57 | 0.10702 | 3.315 |
| our F16 -> our Q8_0 | 0.04121 | 0.84010 | 0.64731 | 1.51 | 0.11562 | 3.236 |
| cstr Q8_0 -> our Q8_0 | 0.03570 | 0.72735 | 0.73548 | 2.77 | 0.10940 | 2.809 |

PCM errors are phase-sensitive: small model changes propagate through the
source generator and can produce large sample-aligned differences despite much
smaller magnitude-spectrum differences. These statistics validate
deterministic execution and expose regressions, but do not by themselves prove
perceptual equivalence. A release-quality evaluation should add several texts
per language plus STOI/PESQ or human listening and ASR transcript checks.

## Reproducing comparisons

Tensor comparison:

```bash
python3 tools/compare_gguf.py reference.gguf candidate.gguf --top 20
```

Fixed-seed model output comparison:

```bash
python3 tools/compare_model_outputs.py \
  --cli build/kokoro-cli \
  --voice models/generated/voices/kokoro-voice-af_heart.gguf \
  --model reference=models/kokoro-82m-f16.gguf \
  --model candidate=models/generated/kokoro-82m-f16.gguf \
  --reference reference \
  --text "The documented Q Eight model works." \
  --seed 42 --output-dir validation/model-output
```

The output tool preserves every WAV and stderr log and writes `report.json`
with reference-relative and all pairwise waveform/spectral metrics.
