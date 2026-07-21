// kokoro-cli.cpp — minimal CLI for kokoro.cpp TTS.
// Loads model + voice, synthesizes text, writes 16-bit WAV.

#include "kokoro.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static bool write_wav(const char *path, const float *pcm, int n_samples,
                      int sample_rate) {
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "error: cannot open '%s' for writing\n", path);
        return false;
    }

    // Convert float to 16-bit PCM
    std::vector<int16_t> buf((size_t)n_samples);
    for (int i = 0; i < n_samples; i++) {
        float v = pcm[i];
        if (v > 1.0f) v = 1.0f;
        if (v < -1.0f) v = -1.0f;
        buf[i] = (int16_t)(v * 32767.0f);
    }

    const int channels = 1;
    const int bits_per_sample = 16;
    const int byte_rate = sample_rate * channels * bits_per_sample / 8;
    const int block_align = channels * bits_per_sample / 8;
    const int data_size = n_samples * block_align;
    const int file_size = 36 + data_size;

    // RIFF header
    fwrite("RIFF", 1, 4, f);
    uint32_t u32 = (uint32_t)file_size;
    fwrite(&u32, 4, 1, f);
    fwrite("WAVE", 1, 4, f);

    // fmt chunk
    fwrite("fmt ", 1, 4, f);
    u32 = 16;
    fwrite(&u32, 4, 1, f);
    uint16_t u16 = 1; // PCM
    fwrite(&u16, 2, 1, f);
    u16 = (uint16_t)channels;
    fwrite(&u16, 2, 1, f);
    u32 = (uint32_t)sample_rate;
    fwrite(&u32, 4, 1, f);
    u32 = (uint32_t)byte_rate;
    fwrite(&u32, 4, 1, f);
    u16 = (uint16_t)block_align;
    fwrite(&u16, 2, 1, f);
    u16 = (uint16_t)bits_per_sample;
    fwrite(&u16, 2, 1, f);

    // data chunk
    fwrite("data", 1, 4, f);
    u32 = (uint32_t)data_size;
    fwrite(&u32, 4, 1, f);
    const bool ok = fwrite(buf.data(), sizeof(int16_t), (size_t)n_samples, f) == (size_t)n_samples;

    if (fclose(f) != 0 || !ok) {
        fprintf(stderr, "error: failed writing '%s'\n", path);
        return false;
    }
    return true;
}

static void usage(const char *prog) {
    fprintf(stderr,
            "Usage: %s [options] -m MODEL -v VOICE \"text to synthesize\"\n"
            "\n"
            "Options:\n"
            "  -m PATH    Path to Kokoro model GGUF (required)\n"
            "  -v PATH    Path to voice pack GGUF (required)\n"
            "  -o PATH    Output WAV path (default: output.wav)\n"
            "  -l LANG    espeak-ng language (default: en-us)\n"
            "  -t N       Number of threads (default: 4)\n"
            "  -s SCALE   Length scale (default: 1.0)\n"
            "  --phonemes Input is already IPA (skip phonemizer)\n"
            "  -h         Show this help\n",
            prog);
}

int main(int argc, char **argv) {
    const char *model_path = nullptr;
    const char *voice_path = nullptr;
    const char *output_path = "output.wav";
    const char *language = "en-us";
    const char *text = nullptr;
    int n_threads = 4;
    float length_scale = 1.0f;
    bool use_phonemes = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-v") == 0 && i + 1 < argc) {
            voice_path = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (strcmp(argv[i], "-l") == 0 && i + 1 < argc) {
            language = argv[++i];
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            length_scale = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--phonemes") == 0) {
            use_phonemes = true;
        } else if (strcmp(argv[i], "-h") == 0 ||
                   strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else if (argv[i][0] != '-') {
            text = argv[i];
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (!model_path || !voice_path || !text) {
        fprintf(stderr, "Error: model (-m), voice (-v), and text are required\n\n");
        usage(argv[0]);
        return 1;
    }

    // Initialize
    struct kokoro_context_params params = kokoro_context_default_params();
    params.n_threads = n_threads;
    params.length_scale = length_scale;
    strncpy(params.espeak_lang, language, sizeof(params.espeak_lang) - 1);

    struct kokoro_context *ctx = kokoro_init_from_file(model_path, params);
    if (!ctx) {
        fprintf(stderr, "Failed to load model: %s\n", model_path);
        return 1;
    }

    if (kokoro_load_voice_pack(ctx, voice_path) != 0) {
        fprintf(stderr, "Failed to load voice pack: %s\n", voice_path);
        kokoro_free(ctx);
        return 1;
    }

    // Synthesize
    int n_samples = 0;
    float *pcm;
    if (use_phonemes) {
        pcm = kokoro_synthesize_phonemes(ctx, text, &n_samples);
    } else {
        pcm = kokoro_synthesize(ctx, text, &n_samples);
    }

    if (!pcm || n_samples <= 0) {
        fprintf(stderr, "Synthesis failed\n");
        kokoro_free(ctx);
        return 1;
    }

    const int sr = kokoro_sample_rate(ctx);
    if (sr <= 0 || !write_wav(output_path, pcm, n_samples, sr)) {
        kokoro_pcm_free(pcm);
        kokoro_free(ctx);
        return 1;
    }
    fprintf(stderr, "Wrote %d samples (%.2f s) to %s\n", n_samples,
            (float)n_samples / sr, output_path);

    kokoro_pcm_free(pcm);
    kokoro_free(ctx);
    return 0;
}
