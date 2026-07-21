// kokoro.cpp — runtime for hexgrad/Kokoro-82M and yl4579/StyleTTS2-
// LJSpeech (same architecture). The full forward pass at synthesis
// time:
//
//   phonemes (IPA, from espeak-ng or --tts-phonemes) → token IDs
//     ↓
//   text_enc (Embedding → 3× Conv1d k=5 + LN + GELU → bidir LSTM)
//                                                        → t_enc [L, 512]
//   bert (custom ALBERT-base, 12 parameter-shared layers) → bert_dur [L, 512]
//                                                        ↓
//   ref_s = voice_pack[L-1, 0, :]   split [pred 0:128 | dec 128:256]
//                                                        ↓
//   ProsodyPredictor (dur_enc 3× alternating LSTM/AdaLN, post-encoder
//   LSTM, dur_proj, shared LSTM, F0/N AdainResBlk1d stacks) →
//   durations + F0 + N
//                                                        ↓
//   align(t_enc, durations) → en [T_frames, 512]
//                                                        ↓
//   iSTFTNet decoder (encode + 4× decode AdainResBlk1d + asr_res +
//   F0_conv + N_conv → Generator with HnNSF source + 2× upsample +
//   noise injection + 6 resblocks averaged + conv_post → 22 channels)
//                                                        ↓
//   iSTFT on CPU (n_fft=20, hop=5, Hann) → 24 kHz audio
//
// The implementation loads GGUF model and voice weights, phonemizes text,
// executes each inference stage through ggml's backend scheduler, and
// reconstructs PCM.

#include "kokoro.h"
#include "phoneme_utils.h"

#include "core/activation.h"
#include "core/align.h"
#include "core/attention.h"
#include "core/conv.h"
#include "core/dac_decoder.h" // core_dac::fastconv_cache (FASTCONV kernel bake)
#include "core/gguf_loader.h"
#include "core/har_async.h"
#include "core/lstm.h"
#include "core/kokoro_env.h"
#include "core/log.h"
#include "core/matmul.h"
#include "core/m_source_host.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#ifdef KOKORO_HAS_VULKAN
#include "ggml-vulkan.h"
#endif
#ifdef KOKORO_HAS_METAL
#include "ggml-metal.h"
#endif
#ifdef KOKORO_HAS_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <future>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <espeak-ng/speak_lib.h>

// ===========================================================================
// Logging infrastructure
// ===========================================================================

static std::mutex g_log_mutex;
static kokoro_log_cb g_log_cb = nullptr;
static void* g_log_ud = nullptr;

void kokoro_log_internal(enum kokoro_log_level level, const char* fmt, ...) {
    char buf[2048];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    size_t len = std::strlen(buf);
    while (len > 0 && (buf[len - 1] == '\n' || buf[len - 1] == '\r'))
        buf[--len] = '\0';

    kokoro_log_cb cb;
    void* user_data;
    {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        cb = g_log_cb;
        user_data = g_log_ud;
    }
    if (cb) {
        cb(level, buf, user_data);
    }
}

#define kokoro_log kokoro_log_internal

// ===========================================================================
// Thread-local error state
// ===========================================================================

static thread_local std::string g_last_error;

static void kokoro_set_error(const char* fmt, ...) {
    char buf[2048];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    g_last_error = buf;
    kokoro_log(KOKORO_LOG_ERROR, "%s", buf);
}

// ===========================================================================
// Version string
// ===========================================================================

#ifndef KOKORO_VERSION
#define KOKORO_VERSION "0.1.0"
#endif
#ifndef KOKORO_GIT_HASH
#define KOKORO_GIT_HASH "unknown"
#endif

static const char g_version_str[] = "kokoro.cpp " KOKORO_VERSION " (" KOKORO_GIT_HASH ")";

namespace {

// ===========================================================================
// Bench instrumentation — `KOKORO_BENCH=1` for per-stage timings.
// ===========================================================================

bool kokoro_bench_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = kokoro_env::get("KOKORO_BENCH");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

using kokoro_bench_clock = std::chrono::steady_clock;

void kokoro_bench_report(const char* name, double ms) {
    if (!kokoro_bench_enabled())
        return;
    kokoro_log(KOKORO_LOG_DEBUG, "kokoro: benchmark %-22s %.2f ms", name, ms);
}

void kokoro_bench_elapsed(const char* name, kokoro_bench_clock::time_point start) {
    if (!kokoro_bench_enabled())
        return;
    const double ms = std::chrono::duration<double, std::milli>(
                          kokoro_bench_clock::now() - start)
                          .count();
    kokoro_bench_report(name, ms);
}

struct kokoro_bench_stage {
    const char* name;
    kokoro_bench_clock::time_point t0;
    explicit kokoro_bench_stage(const char* n) : name(n), t0(kokoro_bench_clock::now()) {}
    ~kokoro_bench_stage() {
        kokoro_bench_elapsed(name, t0);
    }
};

// Hyperparameters read from `kokoro.*` and `kokoro.plbert.*` GGUF KV.
struct kokoro_hp {
    // Top-level
    uint32_t hidden_dim = 512;
    uint32_t style_dim = 128;
    uint32_t max_dur = 50;
    uint32_t n_token = 178;
    uint32_t n_mels = 80; // unused by the synthesis path; kept for completeness
    uint32_t n_layer = 3; // duration-encoder depth
    uint32_t text_enc_k = 5;
    uint32_t sample_rate = 24000;
    uint32_t vocab_size = 178;

    // PL-BERT (custom 12-layer parameter-shared ALBERT)
    uint32_t plbert_embd_size = 128;
    uint32_t plbert_hidden = 768;
    uint32_t plbert_n_layers = 12;
    uint32_t plbert_n_heads = 12;
    uint32_t plbert_ff = 2048;
    uint32_t plbert_max_pos = 512;
    uint32_t plbert_vocab_size = 178;

    // iSTFTNet
    uint32_t istft_init_ch = 512;
    uint32_t istft_n_fft = 20;
    uint32_t istft_hop = 5;
    uint32_t istft_n_dilations = 3;                    // entries per resblock
    std::vector<uint32_t> istft_upsample_rates;        // [10, 6]
    std::vector<uint32_t> istft_upsample_kernel_sizes; // [20, 12]
    std::vector<uint32_t> istft_resblock_kernel_sizes; // [3, 7, 11]
    std::vector<uint32_t> istft_resblock_dilations;    // flat, length 3*n_dilations
};

struct kokoro_vocab {
    std::vector<std::string> id_to_token;                 // size = vocab_size
    std::unordered_map<std::string, int32_t> token_to_id; // lookup for tokenizer
    int32_t pad_id = 0;                                   // index of "$" (Kokoro/StyleTTS2 pad)
};

struct kokoro_voice_pack {
    std::string name;
    uint32_t max_phonemes = 0;
    uint32_t style_dim = 0;
    ggml_tensor* pack = nullptr; // (max_phon, 1, 256) F32 — owned by vp_ctx_w/vp_buf_w
    ggml_context* vp_ctx_w = nullptr;
    ggml_backend_buffer_t vp_buf_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;
};

// Bounded LRU for (lang \0 text) → IPA phonemes. Phonemization is the
// wall-clock floor for short inputs, so caching repeated text pays off.
struct kokoro_phoneme_cache {
    static constexpr size_t kMax = 1024;
    using value_t = std::pair<std::string, std::string>; // key, phonemes
    std::list<value_t> lru;
    std::unordered_map<std::string, std::list<value_t>::iterator> idx;
    std::mutex mu;

    bool lookup(const std::string& key, std::string& out) {
        std::lock_guard<std::mutex> g(mu);
        auto it = idx.find(key);
        if (it == idx.end())
            return false;
        lru.splice(lru.begin(), lru, it->second);
        out = it->second->second;
        return true;
    }

    void insert(const std::string& key, const std::string& val) {
        std::lock_guard<std::mutex> g(mu);
        auto it = idx.find(key);
        if (it != idx.end()) {
            it->second->second = val;
            lru.splice(lru.begin(), lru, it->second);
            return;
        }
        lru.emplace_front(key, val);
        idx[key] = lru.begin();
        if (lru.size() > kMax) {
            idx.erase(lru.back().first);
            lru.pop_back();
        }
    }

    void clear() {
        std::lock_guard<std::mutex> g(mu);
        lru.clear();
        idx.clear();
    }
};

} // namespace

struct kokoro_context {
    kokoro_context_params params{};
    int n_threads = 4;
    std::string espeak_lang = "en-us";

    kokoro_hp hp;
    kokoro_vocab vocab;

    // Primary compute backend plus CPU fallback for unsupported operations.
    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    enum kokoro_backend selected_backend = KOKORO_BACKEND_CPU;

    // Scheduler for all graph compute.
    ggml_backend_sched_t sched = nullptr;
    ggml_backend_sched_t gen_sched = nullptr;

    // Primary weights (talker GGUF). Tensors stay resident here for the
    // life of the context.
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    // Pre-permuted ConvTranspose1d weights for decomposed mul_mat + col2im_1d.
    ggml_tensor* ups_w_perm[2] = {nullptr, nullptr}; // gen.ups.{0,1}
    ggml_context* ctx_perm = nullptr;
    ggml_backend_buffer_t buf_perm = nullptr;
    std::vector<uint8_t> compute_meta;

    // FASTCONV: baked F32 copies of the F16 conv kernels (cast-kill). Owns its own
    // ctx+buffer; freed in kokoro_free before the backend.
    core_dac::fastconv_cache fc;

    // Voice pack (secondary GGUF).
    kokoro_voice_pack vp;
    bool vp_loaded = false;

    // Phoneme cache (lang \0 text → IPA). Lives for the life of the context.
    kokoro_phoneme_cache phon_cache;

    // Host-cached m_source Linear(9→1) weights. Harmonic generation must not
    // call ggml_backend_tensor_get while the Vulkan scheduler is computing.
    float m_source_weight[core_m_source::kDim] = {};
    float m_source_bias = 0.0f;
    bool m_source_cached = false;
};

namespace {

ggml_tensor* try_get(const kokoro_context* c, const char* name) {
    auto it = c->tensors.find(name);
    return it == c->tensors.end() ? nullptr : it->second;
}

ggml_tensor* require(const kokoro_context* c, const char* name) {
    auto* t = try_get(c, name);
    if (!t) {
        kokoro_set_error("required tensor missing: %s", name);
    }
    return t;
}

// Cache m_source Linear(9→1) weight/bias as host floats during init. Harmonic
// generation later must not call ggml_backend_tensor_get concurrently with
// scheduler compute on Vulkan.
static bool kokoro_cache_m_source(kokoro_context* c) {
    ggml_tensor* lw = require(c, "dec.gen.m_source.weight");
    ggml_tensor* lb = require(c, "dec.gen.m_source.bias");
    if (!lw || !lb)
        return false;
    if (lw->ne[0] != core_m_source::kDim) {
        kokoro_set_error("m_source.weight ne[0]=%lld want %d", (long long)lw->ne[0], core_m_source::kDim);
        return false;
    }

    if (lw->type == GGML_TYPE_F16) {
        uint16_t tmp[core_m_source::kDim];
        ggml_backend_tensor_get(lw, tmp, 0, sizeof(tmp));
        ggml_fp16_to_fp32_row((const ggml_fp16_t*)tmp, c->m_source_weight, core_m_source::kDim);
    } else if (lw->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(lw, c->m_source_weight, 0, sizeof(float) * (size_t)core_m_source::kDim);
    } else {
        kokoro_set_error("m_source.weight unsupported type %d", (int)lw->type);
        return false;
    }

    if (lb->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(lb, &c->m_source_bias, 0, sizeof(float));
    } else if (lb->type == GGML_TYPE_F16) {
        ggml_fp16_t tmp;
        ggml_backend_tensor_get(lb, &tmp, 0, sizeof(tmp));
        c->m_source_bias = ggml_fp16_to_fp32(tmp);
    } else {
        kokoro_set_error("m_source.bias unsupported type %d", (int)lb->type);
        return false;
    }

    c->m_source_cached = true;
    return true;
}

// Read a uint32 array from GGUF metadata into a std::vector. Empty result if
// the key is absent or the array element type is unexpected.
std::vector<uint32_t> kv_u32_array(gguf_context* g, const char* key) {
    std::vector<uint32_t> out;
    const int k = gguf_find_key(g, key);
    if (k < 0)
        return out;
    if (gguf_get_kv_type(g, k) != GGUF_TYPE_ARRAY)
        return out;
    const gguf_type elem = gguf_get_arr_type(g, k);
    const int n = gguf_get_arr_n(g, k);
    out.reserve((size_t)n);
    if (elem == GGUF_TYPE_UINT32 || elem == GGUF_TYPE_INT32) {
        const auto* d = (const uint32_t*)gguf_get_arr_data(g, k);
        out.assign(d, d + n);
    } else if (elem == GGUF_TYPE_UINT64 || elem == GGUF_TYPE_INT64) {
        const auto* d = (const uint64_t*)gguf_get_arr_data(g, k);
        for (int i = 0; i < n; i++)
            out.push_back((uint32_t)d[i]);
    }
    return out;
}

// Sanity-check the loaded weights. Soft-validates a representative
// subset rather than enumerating every tensor (the full list is 459
// names; load_weights already provides the full map).
bool sanity_check_weights(const kokoro_context* c) {
    const char* must_have[] = {
        "bert.embd.tok.weight",       "bert.embd_proj.weight",
        "bert.attn_q.weight",         "bert.attn_q.bias",
        "bert.ffn_up.weight",         "bert.pooler.weight",
        "bert_proj.weight",           "bert_proj.bias",
        "text_enc.embd.weight",       "text_enc.cnn.0.conv.weight",
        "text_enc.lstm.weight_ih_l0", "text_enc.lstm.weight_ih_l0_reverse",
        "pred.F0_proj.weight",        "pred.N_proj.weight",
        "dec.gen.conv_post.weight",
    };
    bool ok = true;
    for (const char* name : must_have) {
        if (c->tensors.find(name) == c->tensors.end()) {
            kokoro_set_error("required tensor missing: %s", name);
            ok = false;
        }
    }
    return ok;
}

// ---------------------------------------------------------------------------
// M3 — BERT (PL-BERT, parameter-shared 12-layer ALBERT)
//
// Architecture (from kokoro/custom_albert.py + the AlbertConfig defaults
// inherited by PL-BERT):
//   * Embeddings: tok(178→128) + pos(512→128) + tt(2→128, idx 0)  → LN
//   * embd_proj: Linear 128 → 768
//   * 12× shared layers (post-norm):
//       y = LN(x + self_attn(x), attn_ln, eps=1e-12)
//       y = LN(y + GELU_new(ffn_up(y))→ffn_down, ffn_ln, eps=1e-12)
//   * (pooler exists in the GGUF but is unused by the synthesis path —
//     Kokoro reads `last_hidden_state`, not `pooler_output`.)
//   * bert_proj: Linear 768 → 512, applied per-token.
//
// The "bert_pooler_out" stage name in kokoro.h is legacy from the plan;
// the value at this stage is the per-token last_hidden_state of shape
// (768, L), not a pooled vector. Kept for ABI stability.
// ---------------------------------------------------------------------------

static const float kBertLayerNormEps = 1e-12f;

// Append BERT ops into an existing graph/context. Inputs: int32 "ids" and
// "positions" (length L). Names "bert_pooler_out" (768, L) and
// "bert_proj_out" (512, L). Does not mark them as graph outputs — the
// synthesis path keeps bert_proj_out as a device-resident functional edge
// into the duration predictor.
static ggml_tensor* kokoro_graph_append_bert(ggml_context* ctx0, kokoro_context* c, int L) {
    const auto& hp = c->hp;
    const int D_emb = (int)hp.plbert_embd_size; // 128
    const int D = (int)hp.plbert_hidden;        // 768
    const int n_h = (int)hp.plbert_n_heads;     // 12
    const int n_lay = (int)hp.plbert_n_layers;  // 12 shared
    const int hd = D / n_h;                     // 64
    const float attn_scale = 1.0f / std::sqrt((float)hd);

    ggml_tensor* ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, L);
    ggml_set_name(ids, "ids");
    ggml_set_input(ids);

    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, L);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // ---- Embeddings ----
    ggml_tensor* tok_w = require(c, "bert.embd.tok.weight");
    ggml_tensor* pos_w = require(c, "bert.embd.pos.weight");
    ggml_tensor* tt_w = require(c, "bert.embd.tt.weight");
    ggml_tensor* eln_w = require(c, "bert.embd.ln.weight");
    ggml_tensor* eln_b = require(c, "bert.embd.ln.bias");
    ggml_tensor* ep_w = require(c, "bert.embd_proj.weight");
    ggml_tensor* ep_b = require(c, "bert.embd_proj.bias");

    ggml_tensor* tok_emb = ggml_get_rows(ctx0, tok_w, ids);       // (D_emb, L) F32
    ggml_tensor* pos_emb = ggml_get_rows(ctx0, pos_w, positions); // (D_emb, L) F32
    // token_type_ids defaults to zero — slice tt_w[:, 0:1] and broadcast on L.
    ggml_tensor* tt_view = ggml_view_2d(ctx0, tt_w, D_emb, 1, tt_w->nb[1], 0); // (D_emb, 1) F16
    ggml_tensor* tt_emb = ggml_cast(ctx0, tt_view, GGML_TYPE_F32);             // (D_emb, 1) F32

    ggml_tensor* x = ggml_add(ctx0, tok_emb, pos_emb);
    x = ggml_add(ctx0, x, tt_emb);

    // Embedding LayerNorm (eps=1e-12).
    x = ggml_norm(ctx0, x, kBertLayerNormEps);
    x = ggml_mul(ctx0, x, eln_w);
    x = ggml_add(ctx0, x, eln_b);

    // embd_proj: Linear D_emb → D
    x = core_ggml::mul_mat(ctx0, ep_w, x);
    x = ggml_add(ctx0, x, ep_b); // (D, L)

    // ---- 12 shared ALBERT layers ----
    ggml_tensor* q_w = require(c, "bert.attn_q.weight");
    ggml_tensor* q_b = require(c, "bert.attn_q.bias");
    ggml_tensor* k_w = require(c, "bert.attn_k.weight");
    ggml_tensor* k_b = require(c, "bert.attn_k.bias");
    ggml_tensor* v_w = require(c, "bert.attn_v.weight");
    ggml_tensor* v_b = require(c, "bert.attn_v.bias");
    ggml_tensor* o_w = require(c, "bert.attn_o.weight");
    ggml_tensor* o_b = require(c, "bert.attn_o.bias");
    ggml_tensor* aln_w = require(c, "bert.attn_ln.weight");
    ggml_tensor* aln_b = require(c, "bert.attn_ln.bias");
    ggml_tensor* fu_w = require(c, "bert.ffn_up.weight");
    ggml_tensor* fu_b = require(c, "bert.ffn_up.bias");
    ggml_tensor* fd_w = require(c, "bert.ffn_down.weight");
    ggml_tensor* fd_b = require(c, "bert.ffn_down.bias");
    ggml_tensor* fln_w = require(c, "bert.ffn_ln.weight");
    ggml_tensor* fln_b = require(c, "bert.ffn_ln.bias");

    core_attn::EncoderSelfAttnParams eap{};
    eap.n_heads = n_h;
    eap.n_kv_heads = n_h; // MHA
    eap.head_dim = hd;
    eap.n_kv_grp = 1;
    eap.attn_scale = attn_scale;
    eap.n_ctx_orig = 0;
    eap.rope_theta = 0.0f;
    eap.permute_cont = true;

    for (int il = 0; il < n_lay; il++) {
        // Self-attention with biased Q/K/V/O, no RoPE, no mask.
        ggml_tensor* attn_out = core_attn::encoder_self_attn(ctx0, x, q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b,
                                                             /*positions*/ nullptr, /*mask*/ nullptr, eap);
        x = ggml_add(ctx0, x, attn_out);
        x = ggml_norm(ctx0, x, kBertLayerNormEps);
        x = ggml_mul(ctx0, x, aln_w);
        x = ggml_add(ctx0, x, aln_b);

        // FFN: up → GELU(tanh approx, "gelu_new") → down (with biases).
        ggml_tensor* ffn = core_ggml::mul_mat(ctx0, fu_w, x);
        ffn = ggml_add(ctx0, ffn, fu_b);
        ffn = ggml_gelu(ctx0, ffn); // tanh-approx == ALBERT "gelu_new"
        ffn = core_ggml::mul_mat(ctx0, fd_w, ffn);
        ffn = ggml_add(ctx0, ffn, fd_b);
        x = ggml_add(ctx0, x, ffn);
        x = ggml_norm(ctx0, x, kBertLayerNormEps);
        x = ggml_mul(ctx0, x, fln_w);
        x = ggml_add(ctx0, x, fln_b);
    }

    // Per-token last_hidden_state — exposed as the
    // "bert_pooler_out" stage (legacy name; not actually pooled).
    ggml_tensor* seq = x; // (D, L)
    seq = ggml_cont(ctx0, seq);
    ggml_set_name(seq, "bert_pooler_out");

    // bert_proj: Linear 768 → 512, applied per-token.
    ggml_tensor* bp_w = require(c, "bert_proj.weight");
    ggml_tensor* bp_b = require(c, "bert_proj.bias");
    ggml_tensor* proj = core_ggml::mul_mat(ctx0, bp_w, seq); // (512, L)
    proj = ggml_add(ctx0, proj, bp_b);
    ggml_set_name(proj, "bert_proj_out");
    return proj;
}



// ---------------------------------------------------------------------------
// AdaLayerNorm (used by the predictor's DurationEncoder).
//
// Reference: kokoro/modules.py AdaLayerNorm (eps=1e-5).
//   gamma, beta = chunk(fc(s), 2)        # fc: Linear(style_dim, 2*C)
//   y = (1 + gamma) * LayerNorm(x) + beta
//
// We compute (1 + γ)·x + β as x + x·γ + β to avoid materialising a "1.0"
// tensor for the (1 + γ) term. Mathematically identical, one fewer
// constant input.
//
// Inputs:
//   x      ne = (C, T)  F32, channel-major (LN normalises over ne[0])
//   style  ne = (sty, 1) F32 — predictor reference vector
//   fc_w   ne = (sty, 2C) F16 — Linear weight
//   fc_b   ne = (2C,) F32 — Linear bias
//
// Output: (C, T) F32.
static const float kAdaLnEps = 1e-5f;

static inline ggml_tensor* kokoro_adaln(ggml_context* ctx, ggml_tensor* x, ggml_tensor* style, ggml_tensor* fc_w,
                                        ggml_tensor* fc_b) {
    const int C = (int)x->ne[0];
    // h = fc(s) → (2C, 1)
    ggml_tensor* h = core_ggml::mul_mat(ctx, fc_w, style);
    h = ggml_add(ctx, h, fc_b);
    // chunk along ne[0]: γ ∈ [0, C), β ∈ [C, 2C).
    const size_t ts = ggml_type_size(GGML_TYPE_F32);
    ggml_tensor* gamma = ggml_view_2d(ctx, h, C, 1, h->nb[1], (size_t)0 * C * ts);
    ggml_tensor* beta = ggml_view_2d(ctx, h, C, 1, h->nb[1], (size_t)1 * C * ts);

    ggml_tensor* normed = ggml_norm(ctx, x, kAdaLnEps);
    // x*γ + x → broadcast γ (C, 1) over T
    ggml_tensor* x_gamma = ggml_mul(ctx, normed, gamma);
    ggml_tensor* out = ggml_add(ctx, normed, x_gamma);
    out = ggml_add(ctx, out, beta);
    return out;
}

// ---------------------------------------------------------------------------
// M4 — TextEncoder
//
// Reference: kokoro/modules.py TextEncoder.
//   Embedding(178, 512) → 3× Sequential(
//       Conv1d(512, 512, k=5, pad=2),
//       LayerNorm(512)        # over channel dim, ε=1e-5
//       LeakyReLU(0.2),
//       Dropout(0.2)          # no-op at inference
//   )
//   bidirectional LSTM(in=512, hidden=256) → output (B, D=512, T) after
//   the final transpose.
//
// Input: pad-wrapped phoneme ids (length L).
// Output stage `text_enc_out`: ne = (512, L)  F32.
// ---------------------------------------------------------------------------

static const float kTextEncLayerNormEps = 1e-5f;
static const float kTextEncLeakySlope = 0.2f;

// Vulkan packs bidirectional LSTM forward/reverse directions on GGML batch
// dimension ne[2]=2 so recurrent matmuls and gate ops are shared. CPU/Metal/
// CUDA keep the historical dual-unidir path for bit-identity. KOKORO_LSTM_BATCH=0
// disables the batched path for A/B measurement.
static inline bool kokoro_lstm_batch_dirs(const kokoro_context* c) {
    return c->selected_backend == KOKORO_BACKEND_VULKAN &&
           (kokoro_env::get("KOKORO_LSTM_BATCH") == nullptr || kokoro_env::truthy("KOKORO_LSTM_BATCH"));
}

static ggml_cgraph* kokoro_build_graph_text_enc(kokoro_context* c, int L) {
    const auto& hp = c->hp;
    const int D = (int)hp.hidden_dim; // 512
    const int H_lstm = D / 2;         // 256 (bidir → 2*H = D)
    const int K = (int)hp.text_enc_k; // 5

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), /*no_alloc=*/true};
    ggml_context* ctx0 = ggml_init(ip);
    // 3 conv blocks + 1 bidir LSTM at modest L → a few thousand nodes max.
    // For L up to plbert_max_pos + 2 = 514, the LSTM dominates: 2 directions
    // * L * ~15 ops/step ≈ 15400 nodes. 32k headroom.
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 32768, false);

    ggml_tensor* ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, L);
    ggml_set_name(ids, "ids");
    ggml_set_input(ids);

    // ---- Embedding: (178, 512) → (D, L) ----
    ggml_tensor* emb_w = require(c, "text_enc.embd.weight");
    ggml_tensor* x = ggml_get_rows(ctx0, emb_w, ids); // (D, L) F32

    // ---- 3× Conv1d(k=5, pad=2) + LN(channel) + LeakyReLU(0.2) ----
    // Layout flow per block:
    //   in:        (D, L)         ne[0]=D, ne[1]=L
    //   transpose: (L, D)         for ggml_conv_1d which expects (T, C_in)
    //   conv1d:    (L, D, 1)      output is 3D (OL, OC, B=1)
    //   add bias:  bias reshaped (1, D, 1) broadcasts on L and B
    //   reshape:   (L, D)         drop trailing 1
    //   transpose: (D, L)         channel-major for ggml_norm
    //   norm:      (D, L)         normalises over ne[0]=D
    //   * gamma + bias beta:      (D, L)
    //   leaky:     (D, L)
    auto bias_1d = [&](ggml_tensor* b) { return ggml_reshape_3d(ctx0, b, 1, b->ne[0], 1); };

    for (int il = 0; il < 3; il++) {
        char nm[128];
        std::snprintf(nm, sizeof(nm), "text_enc.cnn.%d.conv.weight", il);
        ggml_tensor* cw = require(c, nm);
        std::snprintf(nm, sizeof(nm), "text_enc.cnn.%d.conv.bias", il);
        ggml_tensor* cb = require(c, nm);
        std::snprintf(nm, sizeof(nm), "text_enc.cnn.%d.ln.gamma", il);
        ggml_tensor* lg = require(c, nm);
        std::snprintf(nm, sizeof(nm), "text_enc.cnn.%d.ln.beta", il);
        ggml_tensor* lb = require(c, nm);

        // (D, L) → (L, D) for conv input layout.
        x = ggml_cont(ctx0, ggml_transpose(ctx0, x));                       // (L, D)
        x = ggml_conv_1d(ctx0, cw, x, /*s=*/1, /*p=*/(K - 1) / 2, /*d=*/1); // (L, D, 1)
        x = ggml_add(ctx0, x, bias_1d(cb));
        // Drop the trailing batch dim → (L, D).
        x = ggml_reshape_2d(ctx0, x, L, D);

        // (L, D) → (D, L) for channel-major LN.
        x = ggml_cont(ctx0, ggml_transpose(ctx0, x)); // (D, L)
        x = ggml_norm(ctx0, x, kTextEncLayerNormEps);
        x = ggml_mul(ctx0, x, lg);
        x = ggml_add(ctx0, x, lb);
        x = ggml_leaky_relu(ctx0, x, kTextEncLeakySlope, /*inplace=*/false);
    }

    // ---- Bidirectional LSTM ----
    // Input shape requirement: (in, T) = (D=512, L). x is already (D, L). ✓
    ggml_tensor* W_ih_f = require(c, "text_enc.lstm.weight_ih_l0");
    ggml_tensor* W_hh_f = require(c, "text_enc.lstm.weight_hh_l0");
    ggml_tensor* b_ih_f = require(c, "text_enc.lstm.bias_ih_l0");
    ggml_tensor* b_hh_f = require(c, "text_enc.lstm.bias_hh_l0");
    ggml_tensor* W_ih_r = require(c, "text_enc.lstm.weight_ih_l0_reverse");
    ggml_tensor* W_hh_r = require(c, "text_enc.lstm.weight_hh_l0_reverse");
    ggml_tensor* b_ih_r = require(c, "text_enc.lstm.bias_ih_l0_reverse");
    ggml_tensor* b_hh_r = require(c, "text_enc.lstm.bias_hh_l0_reverse");

    ggml_tensor* lstm_out =
        core_lstm::lstm_bidir(ctx0, x, W_ih_f, W_hh_f, b_ih_f, b_hh_f, W_ih_r, W_hh_r, b_ih_r, b_hh_r,
                              H_lstm, kokoro_lstm_batch_dirs(c)); // (2H = D, L)

    ggml_set_name(lstm_out, "text_enc_out");
    ggml_set_output(lstm_out);
    ggml_build_forward_expand(gf, lstm_out);

    ggml_free(ctx0);
    return gf;
}



// ---------------------------------------------------------------------------
// M6 — Alignment
//
// Reference (model.py:110-114):
//   indices = repeat_interleave(arange(L), pred_dur)
//   en = d[:, indices]                  # (640, T_frames)
//
// Durations are still rounded on the host (dynamic T_frames). The host
// builds a compact I32 index vector of length T_frames; F0/N and decoder
// graphs then gather with ggml_get_rows so the large (D, T_frames)
// expansions are never materialised on the CPU or uploaded over PCIe.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// AdaIN1d (used by AdainResBlk1d in M5b predictor F0/N stacks and M7
// decoder).
//
// Reference: kokoro/istftnet.py AdaIN1d.
//   InstanceNorm1d(C, affine=True) but the affine γ'/β' were *not* trained
//   meaningfully (kokoro author note), so the converter dropped them. We
//   only have fc.weight (sty, 2C) and fc.bias (2C,).
//
// Math:
//   gamma, beta = chunk(fc(s), 2)
//   y = (1 + gamma) * InstanceNorm1d(x) + beta
//
// InstanceNorm1d on (C, T) layout: per-channel mean/var across T. ggml_norm
// normalises along ne[0]; with (T, C) layout that's per-channel-along-T,
// which is exactly instance norm 1D. So we transpose, norm, transpose back.
// ---------------------------------------------------------------------------

static const float kAdaIn1dEps = 1e-5f; // PyTorch InstanceNorm1d default

static inline ggml_tensor* kokoro_adain1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* style, ggml_tensor* fc_w,
                                          ggml_tensor* fc_b, const char* dbg_prefix = nullptr) {
    const int C = (int)x->ne[0];

    // Instance norm: transpose (C, T) → (T, C); ggml_norm normalises
    // along ne[0]=T per other dim ⇒ per-channel mean+var along T;
    // transpose back. NOTE: ggml-metal's kernel_norm_fuse_impl had a
    // cross-simdgroup reduction bug that produced wrong per-row mean
    // and variance for short T (specifically T values where the last
    // simdgroup ended up with ≤ a few active threads in the prior
    // parallel-sum loop, e.g. T=65 in the kokoro AdaIN1d). The bug
    // cascaded through AdaIN → conv → AdaIN into garbage audio for
    // short utterances ("hello world"). Fixed by a ggml patch
    // in ggml/src/ggml-metal/ggml-metal.metal (search for
    // "serial reduction by thread 0") — that patch MUST stay
    // co-versioned with this code; if you bump ggml without
    // re-applying it, kokoro short-input audio on Metal regresses.
    // See tests/test_metal_norm_repro.cpp for a standalone repro.
    ggml_tensor* xt = ggml_cont(ctx, ggml_transpose(ctx, x));
    if (dbg_prefix) {
        char nm[128];
        std::snprintf(nm, sizeof(nm), "%s_pre_norm_TC", dbg_prefix);
        ggml_set_name(xt, nm);
        ggml_set_output(xt);
    }
    xt = ggml_norm(ctx, xt, kAdaIn1dEps);
    if (dbg_prefix) {
        char nm[128];
        std::snprintf(nm, sizeof(nm), "%s_post_norm_TC", dbg_prefix);
        ggml_set_name(xt, nm);
        ggml_set_output(xt);
    }
    ggml_tensor* normed = ggml_cont(ctx, ggml_transpose(ctx, xt)); // (C, T)
    if (dbg_prefix) {
        char nm[128];
        std::snprintf(nm, sizeof(nm), "%s_normed", dbg_prefix);
        ggml_set_name(normed, nm);
        ggml_set_output(normed);
    }

    // gamma, beta from fc(s).
    ggml_tensor* h = core_ggml::mul_mat(ctx, fc_w, style);
    h = ggml_add(ctx, h, fc_b);
    if (dbg_prefix) {
        char nm[128];
        std::snprintf(nm, sizeof(nm), "%s_h", dbg_prefix);
        ggml_set_name(h, nm);
        ggml_set_output(h);
    }
    const size_t ts = ggml_type_size(GGML_TYPE_F32);
    ggml_tensor* gamma = ggml_view_2d(ctx, h, C, 1, h->nb[1], (size_t)0 * C * ts);
    ggml_tensor* beta = ggml_view_2d(ctx, h, C, 1, h->nb[1], (size_t)1 * C * ts);

    // (1 + γ) * normed + β  →  normed + normed*γ + β  (saves the "1" tensor).
    ggml_tensor* x_gamma = ggml_mul(ctx, normed, gamma);
    if (dbg_prefix) {
        char nm[128];
        std::snprintf(nm, sizeof(nm), "%s_xgamma", dbg_prefix);
        ggml_set_name(x_gamma, nm);
        ggml_set_output(x_gamma);
    }
    ggml_tensor* out = ggml_add(ctx, normed, x_gamma);
    if (dbg_prefix) {
        char nm[128];
        std::snprintf(nm, sizeof(nm), "%s_normed_plus_xgamma", dbg_prefix);
        ggml_set_name(out, nm);
        ggml_set_output(out);
    }
    out = ggml_add(ctx, out, beta);
    if (dbg_prefix) {
        char nm[128];
        std::snprintf(nm, sizeof(nm), "%s_out", dbg_prefix);
        ggml_set_name(out, nm);
        ggml_set_output(out);
    }
    return out;
}

// Thin alias for the depthwise ConvTranspose1d pool layer used by
// every Kokoro AdainResBlk1d with `upsample=True` (predictor F0[1] /
// N[1] and decoder.decode[3]). See core_convt::convt1d_depthwise_2x_k3
// for the (k=3, s=2, p=1, op=1) derivation, including the wrong-end
// trap caught during validation.
static inline ggml_tensor* kokoro_pool_2x_depthwise(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w_kernel,
                                                    ggml_tensor* w_bias) {
    return core_convt::convt1d_depthwise_2x_k3(ctx, x, w_kernel, w_bias);
}

// ---------------------------------------------------------------------------
// AdainResBlk1d (predictor F0/N stacks).
//
// Reference: kokoro/istftnet.py AdainResBlk1d (note: M7 decoder uses a
// *different* class AdaINResBlock1 which has Snake-α; this one uses
// LeakyReLU(0.2)).
//
// Forward:
//   residual = x
//   x' = norm1(residual)              # AdaIN1d on dim_in
//   x' = LeakyReLU(0.2)(x')
//   x' = pool(x')                     # ConvTranspose1d depthwise 2× if upsample, else identity
//   x' = conv1(x')                    # Conv1d(dim_in→dim_out, k=3, pad=1)
//   x' = norm2(x')                    # AdaIN1d on dim_out
//   x' = LeakyReLU(0.2)(x')
//   x' = conv2(x')                    # Conv1d(dim_out→dim_out, k=3, pad=1)
//   shortcut = upsample(residual)     # F.interpolate(2x, nearest) if upsample
//   shortcut = conv1x1(shortcut)      # if learned_sc (dim_in != dim_out)
//   out = (x' + shortcut) / sqrt(2)
//
// Pass nullptr for `pool_w/pool_b` to skip pool (upsample=False blocks),
// and nullptr for `conv1x1_w` to skip the learned shortcut conv (when
// dim_in == dim_out).
// ---------------------------------------------------------------------------

static const float kAdainLeakySlope = 0.2f;

static inline ggml_tensor* kokoro_adain_resblk(ggml_context* ctx, ggml_tensor* x, ggml_tensor* style,
                                               ggml_tensor* adain1_w, ggml_tensor* adain1_b, ggml_tensor* adain2_w,
                                               ggml_tensor* adain2_b, ggml_tensor* conv1_w, ggml_tensor* conv1_b,
                                               ggml_tensor* conv2_w, ggml_tensor* conv2_b, ggml_tensor* pool_w,
                                               ggml_tensor* pool_b, ggml_tensor* conv1x1_w,
                                               const char* dbg_prefix = nullptr) {
    const bool upsample = (pool_w != nullptr);
    const int dim_in = (int)x->ne[0];
    (void)dim_in;

    // ---- Residual path ----
    char nm[128];
    auto tag = [&](ggml_tensor* t, const char* suffix) {
        if (!dbg_prefix)
            return;
        std::snprintf(nm, sizeof(nm), "%s_%s", dbg_prefix, suffix);
        ggml_set_name(t, nm);
        ggml_set_output(t);
    };
    char ad1_prefix_buf[80];
    const char* ad1_prefix = nullptr;
    if (dbg_prefix) {
        std::snprintf(ad1_prefix_buf, sizeof(ad1_prefix_buf), "%s_adain1", dbg_prefix);
        ad1_prefix = ad1_prefix_buf;
    }
    ggml_tensor* xt = kokoro_adain1d(ctx, x, style, adain1_w, adain1_b, ad1_prefix); // (Cin, T)
    xt = ggml_leaky_relu(ctx, xt, kAdainLeakySlope, /*inplace=*/false);
    tag(xt, "after_lr1");

    if (upsample) {
        xt = kokoro_pool_2x_depthwise(ctx, xt, pool_w, pool_b); // (Cin, 2T)
        tag(xt, "after_pool");
    }

    // conv1: Conv1d(dim_in → dim_out, k=3, pad=1). Layout flow:
    //   (Cin, T') → transpose → (T', Cin) → ggml_conv_1d → (T', Cout, 1) → squeeze → (T', Cout) → transpose → (Cout, T')
    auto conv_k3 = [&](ggml_tensor* in, ggml_tensor* w, ggml_tensor* b) -> ggml_tensor* {
        const int Tin = (int)in->ne[1];
        ggml_tensor* y = ggml_cont(ctx, ggml_transpose(ctx, in)); // (T, Cin)
        y = ggml_conv_1d(ctx, w, y, /*s*/ 1, /*p*/ 1, /*d*/ 1);   // (T, Cout, 1)
        // Add bias broadcast: bias ne=(Cout,) → reshape (1, Cout, 1)
        ggml_tensor* b3 = ggml_reshape_3d(ctx, b, 1, b->ne[0], 1);
        y = ggml_add(ctx, y, b3);
        const int Cout = (int)w->ne[2];
        y = ggml_reshape_2d(ctx, y, Tin, Cout);        // (T, Cout)
        return ggml_cont(ctx, ggml_transpose(ctx, y)); // (Cout, T)
    };

    xt = conv_k3(xt, conv1_w, conv1_b); // (Cout, T')
    tag(xt, "after_conv1");
    char ad2_prefix_buf[80];
    const char* ad2_prefix = nullptr;
    if (dbg_prefix) {
        std::snprintf(ad2_prefix_buf, sizeof(ad2_prefix_buf), "%s_adain2", dbg_prefix);
        ad2_prefix = ad2_prefix_buf;
    }
    xt = kokoro_adain1d(ctx, xt, style, adain2_w, adain2_b, ad2_prefix); // (Cout, T')
    xt = ggml_leaky_relu(ctx, xt, kAdainLeakySlope, /*inplace=*/false);
    tag(xt, "after_lr2");
    xt = conv_k3(xt, conv2_w, conv2_b); // (Cout, T')
    tag(xt, "after_conv2");

    // ---- Shortcut path ----
    ggml_tensor* sc = x;
    if (upsample) {
        // F.interpolate(scale=2, mode='nearest'): each input column is duplicated.
        // Equivalent to (Cin, 1, T) → concat with itself on a new dim → (Cin, 2, T) → reshape (Cin, 2T).
        ggml_tensor* sc_3d = ggml_reshape_3d(ctx, sc, sc->ne[0], 1, sc->ne[1]);
        ggml_tensor* sc_dup = ggml_concat(ctx, sc_3d, sc_3d, /*dim=*/1); // (Cin, 2, T)
        sc = ggml_cont(ctx, ggml_reshape_2d(ctx, sc_dup, sc->ne[0], 2 * (int)sc->ne[1]));
    }
    if (conv1x1_w) {
        // Conv1d k=1, no bias. (Cin, T') → transpose → (T', Cin) → conv → (T', Cout, 1) → reshape → transpose
        const int Tin = (int)sc->ne[1];
        ggml_tensor* sct = ggml_cont(ctx, ggml_transpose(ctx, sc));         // (T', Cin)
        sct = ggml_conv_1d(ctx, conv1x1_w, sct, /*s*/ 1, /*p*/ 0, /*d*/ 1); // (T', Cout, 1)
        const int Cout = (int)conv1x1_w->ne[2];
        sct = ggml_reshape_2d(ctx, sct, Tin, Cout);    // (T', Cout)
        sc = ggml_cont(ctx, ggml_transpose(ctx, sct)); // (Cout, T')
    }

    // out = (xt + sc) / sqrt(2)
    ggml_tensor* sum = ggml_add(ctx, xt, sc);
    return ggml_scale(ctx, sum, 1.0f / std::sqrt(2.0f));
}

// ---------------------------------------------------------------------------
// M5a — ProsodyPredictor (dur_enc + pred.lstm + dur_proj → durations)
//
// Reference: kokoro/modules.py ProsodyPredictor + DurationEncoder.
//
// Voice pack split (corrects the briefing — see model.py:104, 118):
//   ref_s[:, :128]   → DECODER style (used in M7)
//   ref_s[:, 128:]   → PREDICTOR style (used here)
//
// Forward at synth time:
//   bert_proj_out  (D=512, L)         from M3
//   style_pred     (sty=128, 1)        from voice.pack[idx, 0, 128:256]
//
//   x = cat([bert_proj_out, style_pred.broadcast(L)], axis=0)   (640, L)
//   for il in 0..2:
//       x = bidir_LSTM(x)                                        (512, L)
//       x = AdaLN(x, style_pred)                                 (512, L)
//       x = cat([x, style_pred.broadcast(L)], axis=0)            (640, L)
//   dur_enc_out = x                                              (640, L)
//
//   y = bidir_LSTM_pred(x)                                       (512, L)
//   z = Linear_dur_proj(y)                                       (50, L)
//   durations_pre = sum_rows(sigmoid(z))                         (1, L)
//   # ↑ runtime-side: round + clamp(min=1) → int durations[L]
// ---------------------------------------------------------------------------

// Pull the per-voice predictor style vector from the loaded voice pack.
// voice.pack ne = (256, 1, 510) per the converter; index along ne[2] by
// (clamp(L_raw, 1, max_phon) - 1) to get a (256, 1) slice, then take
// the back half (offset 128*sizeof(F32)) for the predictor side.
//
// We bake the index into the graph (graph is rebuilt per-utterance
// anyway), so this returns a graph-time view of the voice-pack tensor.
static ggml_tensor* kokoro_voice_style_slice(ggml_context* ctx, kokoro_context* c, int L_raw, int byte_offset) {
    ggml_tensor* pack = c->vp.pack;
    const uint32_t max_phon = (uint32_t)pack->ne[2];
    int idx = L_raw;
    if (idx < 1)
        idx = 1;
    if (idx > (int)max_phon) {
        kokoro_log(KOKORO_LOG_WARN, "kokoro: phoneme count %d clamped to voice maximum %u", L_raw, max_phon);
        idx = (int)max_phon;
    }
    idx -= 1;
    // Single ggml_view_2d at the combined absolute offset. The previous
    // view-of-view (pack → idx slice → 128-channel half) hit a Metal-
    // specific bug where short utterances on GPU read the wrong slice
    // of the voice pack — the first divergent stages in the per-stage
    // diff (`f0_curve`, `n_curve`) showed amplitude ~15× the PyTorch
    // reference, cascading to dec_encode_out cos=0.27 and
    // dec_decode_3_out cos=-0.30. Folding the two offsets into one
    // view restores the cascade. Note ggml view tensors are named
    // "<parent> (view)" by default — the original logs flagged
    // "voice.pack (view) (view)" which is the symptom of this
    // double-view path.
    return ggml_view_2d(ctx, pack, /*ne0*/ 128, /*ne1*/ 1, pack->nb[1],
                        (size_t)idx * pack->nb[2] + (size_t)byte_offset);
}

// Predictor style: back half (offset 128*F32). model.py:104 → ref_s[:, 128:]
static ggml_tensor* kokoro_voice_style_pred_view(ggml_context* ctx, kokoro_context* c, int L_raw) {
    return kokoro_voice_style_slice(ctx, c, L_raw, 128 * (int)sizeof(float));
}

// Decoder style: front half (offset 0). model.py:118 → ref_s[:, :128]
static ggml_tensor* kokoro_voice_style_dec_view(ggml_context* ctx, kokoro_context* c, int L_raw) {
    return kokoro_voice_style_slice(ctx, c, L_raw, 0);
}

// Append ProsodyPredictor ops into an existing graph/context.
// bert_proj is the functional (512, L) input from BERT's bert_proj_out.
// Required outputs: "dur_enc_out" (640, L) and "durations_pre" (1, L).
static void kokoro_graph_append_predictor(ggml_context* ctx0, ggml_cgraph* gf, kokoro_context* c, int L, int L_raw,
                                          ggml_tensor* bert_proj) {
    const auto& hp = c->hp;
    const int Hsty = (int)hp.style_dim; // 128
    const int H_lstm = (int)hp.hidden_dim / 2; // 256

    // ---- Style: bake voice-pack slice into the graph ----
    // s_pred is F16 (the voice pack is loaded F32 — wait, voice.pack is F32
    // per the GGUF dump — so style is F32 and ggml_mul_mat with the AdaLN
    // fc_w (F16) works fine).
    ggml_tensor* style_pred = kokoro_voice_style_pred_view(ctx0, c, L_raw); // (128, 1) F32

    // Broadcast style to (Hsty, L) for the cat operations.
    ggml_tensor* s_template = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, Hsty, L);
    ggml_tensor* s_broad = ggml_repeat(ctx0, style_pred, s_template); // (128, L)

    // Initial cat([bert_proj, s_broad], dim=0) → (D + Hsty, L) = (640, L)
    ggml_tensor* x = ggml_concat(ctx0, bert_proj, s_broad, /*dim=*/0);

    // ---- 3× alternating bidir-LSTM + AdaLN + cat-with-style ----
    for (int il = 0; il < 3; il++) {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "pred.dur_enc.%d.lstm.weight_ih_l0", il);
        ggml_tensor* W_ih_f = require(c, buf);
        std::snprintf(buf, sizeof(buf), "pred.dur_enc.%d.lstm.weight_hh_l0", il);
        ggml_tensor* W_hh_f = require(c, buf);
        std::snprintf(buf, sizeof(buf), "pred.dur_enc.%d.lstm.bias_ih_l0", il);
        ggml_tensor* b_ih_f = require(c, buf);
        std::snprintf(buf, sizeof(buf), "pred.dur_enc.%d.lstm.bias_hh_l0", il);
        ggml_tensor* b_hh_f = require(c, buf);
        std::snprintf(buf, sizeof(buf), "pred.dur_enc.%d.lstm.weight_ih_l0_reverse", il);
        ggml_tensor* W_ih_r = require(c, buf);
        std::snprintf(buf, sizeof(buf), "pred.dur_enc.%d.lstm.weight_hh_l0_reverse", il);
        ggml_tensor* W_hh_r = require(c, buf);
        std::snprintf(buf, sizeof(buf), "pred.dur_enc.%d.lstm.bias_ih_l0_reverse", il);
        ggml_tensor* b_ih_r = require(c, buf);
        std::snprintf(buf, sizeof(buf), "pred.dur_enc.%d.lstm.bias_hh_l0_reverse", il);
        ggml_tensor* b_hh_r = require(c, buf);

        x = core_lstm::lstm_bidir(ctx0, x, W_ih_f, W_hh_f, b_ih_f, b_hh_f, W_ih_r, W_hh_r, b_ih_r, b_hh_r,
                                  H_lstm, kokoro_lstm_batch_dirs(c)); // (512, L)

        std::snprintf(buf, sizeof(buf), "pred.dur_enc.%d.adaln.weight", il);
        ggml_tensor* fc_w = require(c, buf);
        std::snprintf(buf, sizeof(buf), "pred.dur_enc.%d.adaln.bias", il);
        ggml_tensor* fc_b = require(c, buf);
        x = kokoro_adaln(ctx0, x, style_pred, fc_w, fc_b); // (512, L)

        x = ggml_concat(ctx0, x, s_broad, /*dim=*/0); // (640, L)
    }

    // dur_enc_out — the (640, L) input to pred.lstm and to alignment.
    ggml_tensor* dur_enc_out = ggml_cont(ctx0, x);
    ggml_set_name(dur_enc_out, "dur_enc_out");
    ggml_set_output(dur_enc_out);
    ggml_build_forward_expand(gf, dur_enc_out);

    // ---- pred.lstm: (640, L) → (512, L) ----
    ggml_tensor* pl_W_ih_f = require(c, "pred.lstm.weight_ih_l0");
    ggml_tensor* pl_W_hh_f = require(c, "pred.lstm.weight_hh_l0");
    ggml_tensor* pl_b_ih_f = require(c, "pred.lstm.bias_ih_l0");
    ggml_tensor* pl_b_hh_f = require(c, "pred.lstm.bias_hh_l0");
    ggml_tensor* pl_W_ih_r = require(c, "pred.lstm.weight_ih_l0_reverse");
    ggml_tensor* pl_W_hh_r = require(c, "pred.lstm.weight_hh_l0_reverse");
    ggml_tensor* pl_b_ih_r = require(c, "pred.lstm.bias_ih_l0_reverse");
    ggml_tensor* pl_b_hh_r = require(c, "pred.lstm.bias_hh_l0_reverse");

    ggml_tensor* pl_out =
        core_lstm::lstm_bidir(ctx0, dur_enc_out, pl_W_ih_f, pl_W_hh_f, pl_b_ih_f, pl_b_hh_f, pl_W_ih_r, pl_W_hh_r,
                              pl_b_ih_r, pl_b_hh_r, H_lstm, kokoro_lstm_batch_dirs(c)); // (512, L)
    ggml_set_name(pl_out, "pred_lstm_out");
    ggml_set_output(pl_out);

    // ---- pred.dur_proj: Linear(512 → 50) ----
    ggml_tensor* dp_w = require(c, "pred.dur_proj.weight"); // ne=(512, 50)
    ggml_tensor* dp_b = require(c, "pred.dur_proj.bias");   // ne=(50,)
    ggml_tensor* dp = core_ggml::mul_mat(ctx0, dp_w, pl_out);     // (50, L)
    dp = ggml_add(ctx0, dp, dp_b);
    dp = ggml_sigmoid(ctx0, dp);
    // sum over the 50 dim (ne[0]) → (1, L). Runtime does the round+clamp.
    ggml_tensor* dur_pre = ggml_sum_rows(ctx0, dp); // (1, L)
    ggml_set_name(dur_pre, "durations_pre");
    ggml_set_output(dur_pre);
    ggml_build_forward_expand(gf, dur_pre);
}


// Combined BERT + duration predictor. Inputs: padded token ids/positions.
// bert_proj_out stays device-resident and feeds the predictor as a
// functional edge (no D2H host vector / H2D re-upload). Required outputs:
// dur_enc_out and durations_pre.
static ggml_cgraph* kokoro_build_graph_bert_predictor(kokoro_context* c, int L, int L_raw) {
    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), /*no_alloc=*/true};
    ggml_context* ctx0 = ggml_init(ip);
    // Predictor dominates node count and approaches the old 65k budget at the
    // model's maximum sequence length; leave room for the appended BERT ops.
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 131072, false);

    ggml_tensor* bert_proj = kokoro_graph_append_bert(ctx0, c, L);
    kokoro_graph_append_predictor(ctx0, gf, c, L, L_raw, bert_proj);

    ggml_free(ctx0);
    return gf;
}


// ---------------------------------------------------------------------------
// M5b — F0Ntrain (pred.shared LSTM + F0/N AdainResBlk1d × 3 + F0/N_proj)
//
// Reference: kokoro/modules.py ProsodyPredictor.F0Ntrain.
//   x, _ = self.shared(en.transpose(-1,-2))   # bidir LSTM 640 → 512
//   F0 = x.transpose(-1,-2)                   # (B, 512, T)
//   for block in self.F0:                     # 3 AdainResBlk1d:
//       F0 = block(F0, s)                     # F0[1] upsamples 2×
//   F0 = self.F0_proj(F0)                     # Conv1d 256 → 1, k=1
//   # N mirrors F0 with separate weights
//
// Output dims after the F0 stack:
//   F0[0]: (512, T) → (512, T)
//   F0[1]: (512, T) → (256, 2T)   [upsample=True, dim_in=512, dim_out=256]
//   F0[2]: (256, 2T) → (256, 2T)
//   F0_proj: (256, 2T) → (1, 2T)
//
// Final F0 / N curves are 1D length 2*T_frames.
// ---------------------------------------------------------------------------

static ggml_cgraph* kokoro_build_graph_f0n(kokoro_context* c, int T_frames, int L, int L_raw) {
    const auto& hp = c->hp;
    const int D = (int)hp.hidden_dim; // 512
    const int H_lstm = D / 2;         // 256

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), /*no_alloc=*/true};
    ggml_context* ctx0 = ggml_init(ip);
    // shared LSTM at T_frames (~60-1500) is the dominant cost: 2 dirs × T × ~14 ops.
    // 6 AdainResBlk1d (3 F0 + 3 N), each ~25 ops. 1024-node margin.
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 65536, false);

    // ---- Inputs: compact dur_enc (640, L) + I32 gather indices (T_frames) ----
    // ggml_get_rows selects along ne1 of a (ne0=D, ne1=L) matrix, so
    // en[:, j] = dur_enc[:, align_idx[j]] → (640, T_frames).
    ggml_tensor* dur_enc = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, /*ne0=D+sty=*/640, L);
    ggml_set_name(dur_enc, "dur_enc");
    ggml_set_input(dur_enc);

    ggml_tensor* align_idx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T_frames);
    ggml_set_name(align_idx, "align_idx");
    ggml_set_input(align_idx);

    ggml_tensor* en = ggml_get_rows(ctx0, dur_enc, align_idx); // (640, T_frames)

    // ---- Style: bake voice-pack predictor slice ----
    ggml_tensor* style_pred = kokoro_voice_style_pred_view(ctx0, c, L_raw); // (128, 1) F32

    // ---- pred.shared bidir LSTM (640 → 512) ----
    ggml_tensor* sh_W_ih_f = require(c, "pred.shared.weight_ih_l0");
    ggml_tensor* sh_W_hh_f = require(c, "pred.shared.weight_hh_l0");
    ggml_tensor* sh_b_ih_f = require(c, "pred.shared.bias_ih_l0");
    ggml_tensor* sh_b_hh_f = require(c, "pred.shared.bias_hh_l0");
    ggml_tensor* sh_W_ih_r = require(c, "pred.shared.weight_ih_l0_reverse");
    ggml_tensor* sh_W_hh_r = require(c, "pred.shared.weight_hh_l0_reverse");
    ggml_tensor* sh_b_ih_r = require(c, "pred.shared.bias_ih_l0_reverse");
    ggml_tensor* sh_b_hh_r = require(c, "pred.shared.bias_hh_l0_reverse");

    ggml_tensor* shared_out =
        core_lstm::lstm_bidir(ctx0, en, sh_W_ih_f, sh_W_hh_f, sh_b_ih_f, sh_b_hh_f, sh_W_ih_r, sh_W_hh_r, sh_b_ih_r,
                              sh_b_hh_r, H_lstm, kokoro_lstm_batch_dirs(c)); // (512, T_frames)
    // Named output for diagnostics.
    ggml_set_name(shared_out, "pred_shared_out");
    ggml_set_output(shared_out);
    ggml_build_forward_expand(gf, shared_out);

    // Helper to load AdainResBlk1d weights for prefix "pred.X.{idx}".
    auto load_resblk = [&](const char* prefix, int idx, bool has_pool) {
        struct W {
            ggml_tensor* a1w;
            ggml_tensor* a1b;
            ggml_tensor* a2w;
            ggml_tensor* a2b;
            ggml_tensor* c1w;
            ggml_tensor* c1b;
            ggml_tensor* c2w;
            ggml_tensor* c2b;
            ggml_tensor* poolw;
            ggml_tensor* poolb;
            ggml_tensor* sc;
        } w;
        char buf[96];
        std::snprintf(buf, sizeof(buf), "%s.%d.adain1.weight", prefix, idx);
        w.a1w = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.%d.adain1.bias", prefix, idx);
        w.a1b = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.%d.adain2.weight", prefix, idx);
        w.a2w = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.%d.adain2.bias", prefix, idx);
        w.a2b = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.%d.conv1.weight", prefix, idx);
        w.c1w = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.%d.conv1.bias", prefix, idx);
        w.c1b = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.%d.conv2.weight", prefix, idx);
        w.c2w = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.%d.conv2.bias", prefix, idx);
        w.c2b = require(c, buf);
        if (has_pool) {
            std::snprintf(buf, sizeof(buf), "%s.%d.pool.weight", prefix, idx);
            w.poolw = require(c, buf);
            std::snprintf(buf, sizeof(buf), "%s.%d.pool.bias", prefix, idx);
            w.poolb = require(c, buf);
            std::snprintf(buf, sizeof(buf), "%s.%d.conv1x1.weight", prefix, idx);
            w.sc = require(c, buf);
        } else {
            w.poolw = w.poolb = w.sc = nullptr;
        }
        return w;
    };

    // Opt-in op-level bisect for the F0[0] / N[0] AdainResBlk1d. When
    // KOKORO_DEBUG_INTERMEDIATES=1, every sub-op output inside the
    // first block (AdaIN1d → LeakyReLU → Conv1d → AdaIN1d → LeakyReLU
    // → Conv1d → residual) is named "dbg_pred_{f0,n}_0_…" and marked
    // as a graph output so diagnostic stage dumps can
    // write each one to disk for GPU-vs-CPU comparison. Unset (the
    // default) and the block runs with no extra ops or outputs, so
    // production builds pay zero cost. Used to bisect the ggml_norm
    // Metal regression — keep available for the next per-op-level bug.
    static const bool s_dbg = []() {
        const char* v = kokoro_env::get("KOKORO_DEBUG_INTERMEDIATES");
        return v && *v && *v != '0';
    }();
    auto run_stack = [&](const char* prefix, const char* stage_branch, ggml_tensor* in) -> ggml_tensor* {
        // F0/N stacks all share: idx 0 (no upsample, dim 512→512), idx 1 (upsample, 512→256), idx 2 (no upsample, 256→256).
        ggml_tensor* y = in;
        auto w0 = load_resblk(prefix, 0, /*has_pool=*/false);
        auto w1 = load_resblk(prefix, 1, /*has_pool=*/true);
        auto w2 = load_resblk(prefix, 2, /*has_pool=*/false);
        char dbg0_buf[64];
        const char* dbg0 = nullptr;
        if (s_dbg) {
            std::snprintf(dbg0_buf, sizeof(dbg0_buf), "dbg_pred_%s_0", stage_branch);
            dbg0 = dbg0_buf;
        }
        y = kokoro_adain_resblk(ctx0, y, style_pred, w0.a1w, w0.a1b, w0.a2w, w0.a2b, w0.c1w, w0.c1b, w0.c2w, w0.c2b,
                                /*pool*/ nullptr, nullptr, /*conv1x1*/ nullptr, dbg0);
        // Tag each AdainResBlk1d output as `pred_{f0,n}_{k}_out` so
        // diagnostic dumps can compare against reference
        // and pinpoint the first stage that diverges on Metal.
        {
            char nm[32];
            std::snprintf(nm, sizeof(nm), "pred_%s_0_out", stage_branch);
            y = ggml_cont(ctx0, y);
            ggml_set_name(y, nm);
            ggml_set_output(y);
            ggml_build_forward_expand(gf, y);
        }
        y = kokoro_adain_resblk(ctx0, y, style_pred, w1.a1w, w1.a1b, w1.a2w, w1.a2b, w1.c1w, w1.c1b, w1.c2w, w1.c2b,
                                w1.poolw, w1.poolb, w1.sc);
        {
            char nm[32];
            std::snprintf(nm, sizeof(nm), "pred_%s_1_out", stage_branch);
            y = ggml_cont(ctx0, y);
            ggml_set_name(y, nm);
            ggml_set_output(y);
            ggml_build_forward_expand(gf, y);
        }
        y = kokoro_adain_resblk(ctx0, y, style_pred, w2.a1w, w2.a1b, w2.a2w, w2.a2b, w2.c1w, w2.c1b, w2.c2w, w2.c2b,
                                /*pool*/ nullptr, nullptr, /*conv1x1*/ nullptr);
        {
            char nm[32];
            std::snprintf(nm, sizeof(nm), "pred_%s_2_out", stage_branch);
            y = ggml_cont(ctx0, y);
            ggml_set_name(y, nm);
            ggml_set_output(y);
            ggml_build_forward_expand(gf, y);
        }
        return y; // (256, 2*T_frames)
    };

    // F0 stack
    ggml_tensor* F0 = run_stack("pred.F0", "f0", shared_out);
    // F0_proj: Conv1d(256 → 1, k=1).
    ggml_tensor* fp_w = require(c, "pred.F0_proj.weight"); // ne=(1, 256, 1)
    ggml_tensor* fp_b = require(c, "pred.F0_proj.bias");   // ne=(1,)
    {
        const int Tf = (int)F0->ne[1];
        ggml_tensor* y = ggml_cont(ctx0, ggml_transpose(ctx0, F0));  // (T, 256)
        y = ggml_conv_1d(ctx0, fp_w, y, /*s*/ 1, /*p*/ 0, /*d*/ 1);  // (T, 1, 1)
        y = ggml_add(ctx0, y, ggml_reshape_3d(ctx0, fp_b, 1, 1, 1)); // bias broadcast
        F0 = ggml_reshape_2d(ctx0, y, Tf, 1);                        // (T, 1)
        // ggml_squeeze isn't available; just keep as (T, 1) and treat the 1 dim
        // as a no-op channel for the downstream extractor.
    }
    F0 = ggml_cont(ctx0, F0);
    ggml_set_name(F0, "f0_curve");
    ggml_set_output(F0);
    ggml_build_forward_expand(gf, F0);

    // N stack (mirror)
    ggml_tensor* N = run_stack("pred.N", "n", shared_out);
    ggml_tensor* np_w = require(c, "pred.N_proj.weight");
    ggml_tensor* np_b = require(c, "pred.N_proj.bias");
    {
        const int Tf = (int)N->ne[1];
        ggml_tensor* y = ggml_cont(ctx0, ggml_transpose(ctx0, N));
        y = ggml_conv_1d(ctx0, np_w, y, /*s*/ 1, /*p*/ 0, /*d*/ 1);
        y = ggml_add(ctx0, y, ggml_reshape_3d(ctx0, np_b, 1, 1, 1));
        N = ggml_reshape_2d(ctx0, y, Tf, 1);
    }
    N = ggml_cont(ctx0, N);
    ggml_set_name(N, "n_curve");
    ggml_set_output(N);
    ggml_build_forward_expand(gf, N);

    ggml_free(ctx0);
    return gf;
}


// ---------------------------------------------------------------------------
// M7a/M7b — Decoder body (F0_conv + N_conv + asr_res + encode + 4× decode)
//
// Reference: kokoro/istftnet.py Decoder.forward.
//   F0 = self.F0_conv(F0_curve.unsqueeze(1))    # Conv1d 1→1, k=3, s=2, pad=1
//   N  = self.N_conv(N_curve.unsqueeze(1))      # same
//   x  = cat([asr, F0, N], axis=1)              # (514, T_frames)
//   x  = self.encode(x, s_dec)                  # AdainResBlk1d 514→1024
//   asr_res = self.asr_res(asr)                 # Conv1d 512→64, k=1
//   for i in 0..3:
//       x = cat([x, asr_res, F0, N], axis=1)    # → (1090, T)
//       x = self.decode[i](x, s_dec)            # AdainResBlk1d, last has upsample=True
//
// Output dims:
//   dec_encode_out:    (1024, T_frames)
//   dec.decode[0..2]:  (1024, T_frames)
//   dec.decode[3]:     (512, 2*T_frames)        [upsample=True]
//
// All AdainResBlk1d in the decoder use s_dec (voice pack [0:128], NOT
// the predictor's [128:256]).
// ---------------------------------------------------------------------------

static ggml_cgraph* kokoro_build_graph_decoder_body(kokoro_context* c, int T_frames, int L, int L_raw) {
    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), /*no_alloc=*/true};
    ggml_context* ctx0 = ggml_init(ip);
    // Decoder body has no LSTMs, just convs+AdaIN+LeakyReLU. ~50 ops/block × 5 blocks
    // + a few cat/F0_conv/asr_res steps. 16k node budget is generous.
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 32768, false);

    // ---- Inputs (compact text_enc + align indices, F0/N curves) ----
    // text_enc ne=(512, L); align_idx ne=(T_frames,) I32
    // asr = ggml_get_rows(text_enc, align_idx) → (512, T_frames)
    ggml_tensor* text_enc_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 512, L);
    ggml_set_name(text_enc_in, "text_enc");
    ggml_set_input(text_enc_in);
    ggml_tensor* align_idx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T_frames);
    ggml_set_name(align_idx, "align_idx");
    ggml_set_input(align_idx);
    ggml_tensor* asr_in = ggml_get_rows(ctx0, text_enc_in, align_idx); // (512, T_frames)
    // F0_curve ne=(2*T_frames,) → unsqueeze to (1, 2*T_frames) before F0_conv
    ggml_tensor* f0_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, 2 * T_frames);
    ggml_set_name(f0_in, "f0");
    ggml_set_input(f0_in);
    ggml_tensor* n_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, 2 * T_frames);
    ggml_set_name(n_in, "n");
    ggml_set_input(n_in);

    ggml_tensor* style_dec = kokoro_voice_style_dec_view(ctx0, c, L_raw); // (128, 1)

    // ---- F0_conv / N_conv: Conv1d(1, 1, k=3, s=2, pad=1) ----
    auto small_conv1d = [&](ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, int s, int p) {
        // Input x ne=(1, 2T). Conv expects (T, C, 1) layout — we have (1, 2T) which IS (C=1, T=2T) in ggml,
        // but conv_1d wants (T, C). Transpose to (2T, 1).
        const int Tin = (int)x->ne[1];
        ggml_tensor* y = ggml_cont(ctx0, ggml_transpose(ctx0, x)); // (2T, 1)
        y = ggml_conv_1d(ctx0, w, y, s, p, /*d*/ 1);               // (Tout, 1, 1)
        const int Cout = (int)w->ne[2];
        ggml_tensor* b3 = ggml_reshape_3d(ctx0, b, 1, b->ne[0], 1);
        y = ggml_add(ctx0, y, b3);
        const int Tout = (int)y->ne[0];
        y = ggml_reshape_2d(ctx0, y, Tout, Cout);
        (void)Tin;
        return ggml_cont(ctx0, ggml_transpose(ctx0, y)); // (Cout, Tout)
    };
    ggml_tensor* F0_conv_w = require(c, "dec.F0_conv.weight");
    ggml_tensor* F0_conv_b = require(c, "dec.F0_conv.bias");
    ggml_tensor* N_conv_w = require(c, "dec.N_conv.weight");
    ggml_tensor* N_conv_b = require(c, "dec.N_conv.bias");
    ggml_tensor* F0_d = small_conv1d(f0_in, F0_conv_w, F0_conv_b, /*s*/ 2, /*p*/ 1); // (1, T_frames)
    ggml_tensor* N_d = small_conv1d(n_in, N_conv_w, N_conv_b, /*s*/ 2, /*p*/ 1);     // (1, T_frames)

    // ---- cat([asr, F0_d, N_d], axis=0) → (514, T_frames) ----
    ggml_tensor* x = ggml_concat(ctx0, asr_in, F0_d, /*dim=*/0);
    x = ggml_concat(ctx0, x, N_d, /*dim=*/0); // (514, T)

    // ---- encode: AdainResBlk1d(514→1024) ----
    auto load_decoder_resblk = [&](const char* prefix, bool has_pool) {
        struct W {
            ggml_tensor *a1w, *a1b, *a2w, *a2b, *c1w, *c1b, *c2w, *c2b, *poolw, *poolb, *sc;
        } w;
        char buf[96];
        std::snprintf(buf, sizeof(buf), "%s.adain1.weight", prefix);
        w.a1w = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.adain1.bias", prefix);
        w.a1b = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.adain2.weight", prefix);
        w.a2w = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.adain2.bias", prefix);
        w.a2b = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.conv1.weight", prefix);
        w.c1w = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.conv1.bias", prefix);
        w.c1b = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.conv2.weight", prefix);
        w.c2w = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.conv2.bias", prefix);
        w.c2b = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.conv1x1.weight", prefix);
        w.sc = require(c, buf);
        if (has_pool) {
            std::snprintf(buf, sizeof(buf), "%s.pool.weight", prefix);
            w.poolw = require(c, buf);
            std::snprintf(buf, sizeof(buf), "%s.pool.bias", prefix);
            w.poolb = require(c, buf);
        } else {
            w.poolw = w.poolb = nullptr;
        }
        return w;
    };

    {
        auto e = load_decoder_resblk("dec.encode", /*has_pool=*/false);
        x = kokoro_adain_resblk(ctx0, x, style_dec, e.a1w, e.a1b, e.a2w, e.a2b, e.c1w, e.c1b, e.c2w, e.c2b,
                                /*pool*/ nullptr, nullptr, e.sc); // (1024, T)
    }
    ggml_tensor* dec_encode_out = ggml_cont(ctx0, x);
    ggml_set_name(dec_encode_out, "dec_encode_out");
    ggml_set_output(dec_encode_out);
    ggml_build_forward_expand(gf, dec_encode_out);

    // ---- asr_res: Conv1d(512→64, k=1) ----
    ggml_tensor* asr_res_w = require(c, "dec.asr_res.weight");
    ggml_tensor* asr_res_b = require(c, "dec.asr_res.bias");
    ggml_tensor* asr_res_out;
    {
        const int Tin = (int)asr_in->ne[1];
        ggml_tensor* y = ggml_cont(ctx0, ggml_transpose(ctx0, asr_in));  // (T, 512)
        y = ggml_conv_1d(ctx0, asr_res_w, y, /*s*/ 1, /*p*/ 0, /*d*/ 1); // (T, 64, 1)
        ggml_tensor* b3 = ggml_reshape_3d(ctx0, asr_res_b, 1, asr_res_b->ne[0], 1);
        y = ggml_add(ctx0, y, b3);
        y = ggml_reshape_2d(ctx0, y, Tin, 64);                  // (T, 64)
        asr_res_out = ggml_cont(ctx0, ggml_transpose(ctx0, y)); // (64, T)
    }

    // ---- 4× decode AdainResBlk1d, with cat([x, asr_res, F0_d, N_d]) before each ----
    x = dec_encode_out;
    for (int il = 0; il < 4; il++) {
        // cat → (1024 + 64 + 1 + 1, T) = (1090, T)
        ggml_tensor* xc = ggml_concat(ctx0, x, asr_res_out, /*dim=*/0); // (1088, T)
        xc = ggml_concat(ctx0, xc, F0_d, /*dim=*/0);
        xc = ggml_concat(ctx0, xc, N_d, /*dim=*/0); // (1090, T)

        char prefix[64];
        std::snprintf(prefix, sizeof(prefix), "dec.decode.%d", il);
        const bool has_pool = (il == 3);
        auto w = load_decoder_resblk(prefix, has_pool);
        x = kokoro_adain_resblk(ctx0, xc, style_dec, w.a1w, w.a1b, w.a2w, w.a2b, w.c1w, w.c1b, w.c2w, w.c2b, w.poolw,
                                w.poolb, w.sc);
    }
    // After block 3 (upsample=True): x is (512, 2*T_frames).
    ggml_tensor* dec_decode_3_out = ggml_cont(ctx0, x);
    ggml_set_name(dec_decode_3_out, "dec_decode_3_out");
    ggml_set_output(dec_decode_3_out);
    ggml_build_forward_expand(gf, dec_decode_3_out);

    ggml_free(ctx0);
    return gf;
}


// ---------------------------------------------------------------------------
// M7c — Generator (iSTFTNet)
//
// Reference: kokoro/istftnet.py Generator + AdaINResBlock1 + SineGen +
// SourceModuleHnNSF + TorchSTFT.
//
// Forward at synth time:
//   x = dec_decode_3_out                 (512, 2*T_frames)   from M7b
//   F0_curve = pred.F0_proj output        (1, 2*T_frames)    from M5b
//   s_dec  = voice.pack[:, :128]          (128, 1)
//
//   # CPU-side (pre-built before graph), uses random noise + STFT:
//   f0_up = repeat-300×(F0_curve)        (600*T_frames,)
//   har_source = m_source(SineGen(f0_up))(600*T_frames,)
//   har_spec, har_phase = STFT(har_source, n_fft=20, hop=5, hann periodic, center=True)
//   har = cat(har_spec, har_phase, dim=0) (22, T_har) where T_har = 120*T_frames + 1
//
//   for i in 0..1:
//       x = LeakyReLU(0.1)(x)
//       x_source = noise_convs[i](har)                    # k=12 s=6 (i=0); k=1 (i=1)
//       x_source = noise_res[i](x_source, s_dec)           # AdaINResBlock1 with Snake-α
//       x = ups[i](x)                                       # ConvTranspose1d
//       if i == 1: x = reflection_pad(x, (1, 0))           # left-pad 1
//       x = x + x_source
//       x = mean_j(resblocks[i*3+j](x, s_dec) for j in 0..2)
//   x = LeakyReLU(0.01)(x)                                  # default slope, NOT 0.1!
//   x = conv_post(x)                                        # Conv1d 128→22, k=7, p=3
//   spec  = exp(x[:11, :])                                  # NOT raw mag
//   phase = sin(x[11:, :])                                  # NOT raw phase
//
// All Generator AdaINResBlock1 (note capital B!) use Snake-α activation:
//   y = x + (1/α) * sin²(α*x)   with per-channel α (1, C, 1) F16, init=1.
// This is *different* from the predictor / decoder-body's `kokoro_adain_resblk`
// which uses LeakyReLU(0.2).
// ---------------------------------------------------------------------------

// Snake-α activation. Thin alias to core_act::snake_alpha — see that
// header for the (1, C, 1) layout convention and Metal-typing notes.
// Channel-first (C, T) path — kept for CPU bit-identity.
static inline ggml_tensor* kokoro_snake1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* alpha) {
    return core_act::snake_alpha(ctx, x, alpha);
}

// Vulkan generator path keeps activations in contiguous time-first (T, C).
// ggml_conv_1d / ggml_norm / Vulkan's fused Snake kernel are all naturally
// (T, C); staying in that layout removes the CONT transpose tax around every
// AdaIN and Conv1d. Gated to Vulkan only so CPU (and untested Metal/CUDA)
// keep the historical channel-first graph.
static inline bool kokoro_generator_time_first(const kokoro_context* c) {
    return c->selected_backend == KOKORO_BACKEND_VULKAN;
}

// AdaIN1d on contiguous (T, C): ggml_norm already reduces along ne[0]=T.
// gamma/beta are reshaped to (1, C) so they broadcast over time.
static inline ggml_tensor* kokoro_adain1d_tf(ggml_context* ctx, ggml_tensor* x, ggml_tensor* style, ggml_tensor* fc_w,
                                            ggml_tensor* fc_b) {
    const int C = (int)x->ne[1];
    ggml_tensor* normed = ggml_norm(ctx, x, kAdaIn1dEps); // (T, C)

    ggml_tensor* h = core_ggml::mul_mat(ctx, fc_w, style);
    h = ggml_add(ctx, h, fc_b);
    const size_t ts = ggml_type_size(GGML_TYPE_F32);
    ggml_tensor* gamma = ggml_reshape_2d(ctx, ggml_view_1d(ctx, h, C, (size_t)0 * C * ts), 1, C); // (1, C)
    ggml_tensor* beta = ggml_reshape_2d(ctx, ggml_view_1d(ctx, h, C, (size_t)1 * C * ts), 1, C);  // (1, C)

    ggml_tensor* x_gamma = ggml_mul(ctx, normed, gamma);
    ggml_tensor* out = ggml_add(ctx, normed, x_gamma);
    return ggml_add(ctx, out, beta);
}

// Prepared Snake-α broadcast operands for Vulkan fusion.
// Fusion requires exact MUL→SIN→SQR→MUL→ADD with contiguous x=(T,C) and
// F32 a/inv_a=(1,C). The reciprocal is built as small graph ops; callers must
// expand inv_a into the graph *before* the Snake chain so the DIV does not
// sit between SQR and the second MUL (which would break ggml_can_fuse).
struct kokoro_snake_alpha_tf {
    ggml_tensor* a = nullptr;     // (1, C) F32
    ggml_tensor* inv_a = nullptr; // (1, C) F32 = 1/a
};

static inline kokoro_snake_alpha_tf kokoro_snake_alpha_prepare_tf(ggml_context* ctx, ggml_tensor* alpha) {
    const int C = (int)ggml_nelements(alpha);
    ggml_tensor* a = ggml_reshape_2d(ctx, alpha, 1, C);
    a = ggml_cast(ctx, a, GGML_TYPE_F32);                                  // (1, C) F32
    ggml_tensor* ones = ggml_scale_bias(ctx, a, /*s=*/0.0f, /*b=*/1.0f);  // 0·a + 1
    ggml_tensor* inv_a = ggml_div(ctx, ones, a);
    return {a, inv_a};
}

static inline ggml_tensor* kokoro_snake1d_tf(ggml_context* ctx, ggml_tensor* x, const kokoro_snake_alpha_tf& alpha) {
    // y = x + sin²(a·x) · inv_a   — exact Vulkan snake_pattern chain.
    ggml_tensor* ax = ggml_mul(ctx, x, alpha.a);
    ggml_tensor* sin_ax = ggml_sin(ctx, ax);
    ggml_tensor* sin_sq = ggml_sqr(ctx, sin_ax);
    ggml_tensor* scaled = ggml_mul(ctx, sin_sq, alpha.inv_a);
    return ggml_add(ctx, x, scaled);
}

// AdaINResBlock1 (NOTE the capital "B" — this is the Generator's class, not
// the predictor/decoder-body's `kokoro_adain_resblk`).
// 3 sub-blocks (j=0..2) looping over `dilations` for the convs1; convs2
// always uses dilation=1. All convs use kernel_size=K with same-padding.
struct kokoro_resblock1_w {
    int K = 0;
    int dilations[3] = {1, 3, 5};
    ggml_tensor* a1w[3] = {};
    ggml_tensor* a1b[3] = {};
    ggml_tensor* a2w[3] = {};
    ggml_tensor* a2b[3] = {};
    ggml_tensor* alpha1[3] = {};
    ggml_tensor* alpha2[3] = {};
    ggml_tensor* c1w[3] = {};
    ggml_tensor* c1b[3] = {};
    ggml_tensor* c2w[3] = {};
    ggml_tensor* c2b[3] = {};
};

static void kokoro_load_resblock1(const kokoro_context* c, kokoro_resblock1_w& w, const char* prefix, int K,
                                  const int dilations[3]) {
    char buf[128];
    w.K = K;
    for (int j = 0; j < 3; j++)
        w.dilations[j] = dilations[j];
    for (int j = 0; j < 3; j++) {
        std::snprintf(buf, sizeof(buf), "%s.adain1.%d.weight", prefix, j);
        w.a1w[j] = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.adain1.%d.bias", prefix, j);
        w.a1b[j] = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.adain2.%d.weight", prefix, j);
        w.a2w[j] = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.adain2.%d.bias", prefix, j);
        w.a2b[j] = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.alpha1.%d", prefix, j);
        w.alpha1[j] = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.alpha2.%d", prefix, j);
        w.alpha2[j] = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.convs1.%d.weight", prefix, j);
        w.c1w[j] = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.convs1.%d.bias", prefix, j);
        w.c1b[j] = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.convs2.%d.weight", prefix, j);
        w.c2w[j] = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.convs2.%d.bias", prefix, j);
        w.c2b[j] = require(c, buf);
    }
}

// Conv1d helper used by AdaINResBlock1: kernel size K, dilation d, same-padding.
// in: (Cin, T) F32. w: ne=(K, Cin, Cout) F16. b: ne=(Cout,) F32.
// Returns (Cout, T) F32.
static inline ggml_tensor* kokoro_conv1d_kd(ggml_context* ctx, ggml_tensor* in, ggml_tensor* w, ggml_tensor* b, int K,
                                            int d) {
    const int Tin = (int)in->ne[1];
    const int Cout = (int)w->ne[2];
    const int p = d * (K - 1) / 2;                            // same-padding for stride=1
    ggml_tensor* y = ggml_cont(ctx, ggml_transpose(ctx, in)); // (T, Cin)
    y = ggml_conv_1d(ctx, w, y, /*s*/ 1, p, d);               // (T, Cout, 1)
    if (b) {
        ggml_tensor* b3 = ggml_reshape_3d(ctx, b, 1, b->ne[0], 1);
        y = ggml_add(ctx, y, b3);
    }
    y = ggml_reshape_2d(ctx, y, Tin, Cout);        // (T, Cout)
    return ggml_cont(ctx, ggml_transpose(ctx, y)); // (Cout, T)
}

// Time-first Conv1d: in/out stay (T, C). No transpose CONT around the op.
static inline ggml_tensor* kokoro_conv1d_kd_tf(ggml_context* ctx, ggml_tensor* in, ggml_tensor* w, ggml_tensor* b, int K,
                                               int d) {
    const int Tin = (int)in->ne[0];
    const int Cout = (int)w->ne[2];
    const int p = d * (K - 1) / 2;
    ggml_tensor* y = ggml_conv_1d(ctx, w, in, /*s*/ 1, p, d); // (T, Cout, 1)
    if (b) {
        ggml_tensor* b3 = ggml_reshape_3d(ctx, b, 1, b->ne[0], 1);
        y = ggml_add(ctx, y, b3);
    }
    return ggml_reshape_2d(ctx, y, Tin, Cout); // (T, Cout)
}

static inline ggml_tensor* kokoro_resblock1_forward(ggml_context* ctx, ggml_tensor* x, ggml_tensor* style,
                                                    const kokoro_resblock1_w& w) {
    for (int j = 0; j < 3; j++) {
        ggml_tensor* xt = kokoro_adain1d(ctx, x, style, w.a1w[j], w.a1b[j]);
        xt = kokoro_snake1d(ctx, xt, w.alpha1[j]);
        xt = kokoro_conv1d_kd(ctx, xt, w.c1w[j], w.c1b[j], w.K, w.dilations[j]);
        xt = kokoro_adain1d(ctx, xt, style, w.a2w[j], w.a2b[j]);
        xt = kokoro_snake1d(ctx, xt, w.alpha2[j]);
        xt = kokoro_conv1d_kd(ctx, xt, w.c2w[j], w.c2b[j], w.K, /*d=*/1);
        x = ggml_add(ctx, xt, x);
    }
    return x;
}

// Time-first AdaINResBlock1. alpha1/alpha2 must already be prepared and their
// inv_a tensors expanded into `gf` before this runs.
static inline ggml_tensor* kokoro_resblock1_forward_tf(ggml_context* ctx, ggml_tensor* x, ggml_tensor* style,
                                                       const kokoro_resblock1_w& w, const kokoro_snake_alpha_tf alpha1[3],
                                                       const kokoro_snake_alpha_tf alpha2[3]) {
    for (int j = 0; j < 3; j++) {
        ggml_tensor* xt = kokoro_adain1d_tf(ctx, x, style, w.a1w[j], w.a1b[j]);
        xt = kokoro_snake1d_tf(ctx, xt, alpha1[j]);
        xt = kokoro_conv1d_kd_tf(ctx, xt, w.c1w[j], w.c1b[j], w.K, w.dilations[j]);
        xt = kokoro_adain1d_tf(ctx, xt, style, w.a2w[j], w.a2b[j]);
        xt = kokoro_snake1d_tf(ctx, xt, alpha2[j]);
        xt = kokoro_conv1d_kd_tf(ctx, xt, w.c2w[j], w.c2b[j], w.K, /*d=*/1);
        x = ggml_add(ctx, xt, x);
    }
    return x;
}

// PyTorch ConvTranspose1d wrapper: uses decomposed mul_mat + col2im_1d when
// w_perm is available, otherwise falls back to ggml_conv_transpose_1d with
// manual symmetric crop.
//
// in: (Cin, T) F32. w: ne=(K, Cout, Cin) F16. b: ne=(Cout,) F32.
// w_perm: ne=(Cin, K*Cout) F32 or nullptr.
// Returns (Cout, T_out) F32.
static inline ggml_tensor* kokoro_convt1d_pad(ggml_context* ctx, ggml_tensor* in, ggml_tensor* w, ggml_tensor* w_perm,
                                              ggml_tensor* b, int stride, int pad) {
    if (w_perm) {
        const int K = (int)w->ne[0];
        return core_convt::convt1d_decomp(ctx, in, w_perm, b, stride, K, pad, pad);
    }
    // Old path — stable, works on CPU without the col2im op.
    const int Cout = (int)w->ne[1];
    ggml_tensor* xT = ggml_cont(ctx, ggml_transpose(ctx, in));         // (T, Cin)
    ggml_tensor* y = ggml_conv_transpose_1d(ctx, w, xT, stride, 0, 1); // (T_unpad, Cout, 1, 1)
    const int T_unpad = (int)y->ne[0];
    const int T_out = T_unpad - 2 * pad;
    y = ggml_reshape_2d(ctx, y, T_unpad, Cout); // (T_unpad, Cout)
    if (pad > 0) {
        // Slice [pad : pad + T_out] along the time (ne[0]) axis.
        y = ggml_view_2d(ctx, y, T_out, Cout, (size_t)T_unpad * sizeof(float), (size_t)pad * sizeof(float));
        y = ggml_cont(ctx, y); // (T_out, Cout)
    }
    ggml_tensor* yT = ggml_cont(ctx, ggml_transpose(ctx, y)); // (Cout, T_out)
    if (b)
        yT = ggml_add(ctx, yT, b); // bias broadcasts on T
    return yT;
}

// Time-first ConvTranspose1d. Decomposed mul_mat+col2im is still channels-first
// internally, so this pays one conversion at each of its two boundaries via
// convt1d_decomp_tf — do not rewrite GGML here.
static inline ggml_tensor* kokoro_convt1d_pad_tf(ggml_context* ctx, ggml_tensor* in, ggml_tensor* w, ggml_tensor* w_perm,
                                                 ggml_tensor* b, int stride, int pad) {
    if (w_perm) {
        const int K = (int)w->ne[0];
        return core_convt::convt1d_decomp_tf(ctx, in, w_perm, b, stride, K, pad, pad);
    }
    // Fallback without w_perm: ggml_conv_transpose_1d already wants (T, Cin).
    const int Cout = (int)w->ne[1];
    ggml_tensor* y = ggml_conv_transpose_1d(ctx, w, in, stride, 0, 1); // (T_unpad, Cout, 1, 1)
    const int T_unpad = (int)y->ne[0];
    const int T_out = T_unpad - 2 * pad;
    y = ggml_reshape_2d(ctx, y, T_unpad, Cout);
    if (pad > 0) {
        y = ggml_view_2d(ctx, y, T_out, Cout, (size_t)T_unpad * sizeof(float), (size_t)pad * sizeof(float));
        y = ggml_cont(ctx, y); // (T_out, Cout)
    }
    if (b)
        y = ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, (int)b->ne[0]));
    return y;
}

// Conv1d helper for noise_convs[i]. Different from kokoro_conv1d_kd in that K
// and stride may be non-trivial (k=12,s=6 for noise_convs[0]; k=1,s=1 for
// noise_convs[1]). Does PyTorch-style same-output handling via explicit
// padding param.
static inline ggml_tensor* kokoro_conv1d_ks(ggml_context* ctx, ggml_tensor* in, ggml_tensor* w, ggml_tensor* b, int s,
                                            int p) {
    const int Cout = (int)w->ne[2];
    ggml_tensor* y = ggml_cont(ctx, ggml_transpose(ctx, in)); // (T, Cin)
    y = ggml_conv_1d(ctx, w, y, s, p, /*d*/ 1);               // (T_out, Cout, 1)
    if (b) {
        ggml_tensor* b3 = ggml_reshape_3d(ctx, b, 1, b->ne[0], 1);
        y = ggml_add(ctx, y, b3);
    }
    const int Tout = (int)y->ne[0];
    y = ggml_reshape_2d(ctx, y, Tout, Cout);
    return ggml_cont(ctx, ggml_transpose(ctx, y)); // (Cout, T_out)
}

// Time-first noise/post Conv1d: in/out (T, C).
static inline ggml_tensor* kokoro_conv1d_ks_tf(ggml_context* ctx, ggml_tensor* in, ggml_tensor* w, ggml_tensor* b, int s,
                                               int p) {
    const int Cout = (int)w->ne[2];
    ggml_tensor* y = ggml_conv_1d(ctx, w, in, s, p, /*d*/ 1); // (T_out, Cout, 1)
    if (b) {
        ggml_tensor* b3 = ggml_reshape_3d(ctx, b, 1, b->ne[0], 1);
        y = ggml_add(ctx, y, b3);
    }
    const int Tout = (int)y->ne[0];
    return ggml_reshape_2d(ctx, y, Tout, Cout); // (T_out, Cout)
}

// ---------------------------------------------------------------------------
// CPU-side: produce `har` (22, T_har) from f0_curve via SineGen + l_linear +
// tanh + STFT. Has random noise (rand_ini, randn_like) so MUST be CPU.
//
// Reference: istftnet.py SineGen._f02sine + SourceModuleHnNSF.forward +
// TorchSTFT.transform.
//
//   upsample_scale = 300 (= 10 * 6 * 5 from upsample_rates [10, 6] * hop 5)
//   harmonic_num = 8, dim = 9, sample_rate = 24000
//   sine_amp = 0.1, noise_std = 0.003, voiced_threshold = 10
//   n_fft = 20, hop = 5, hann periodic, center=True
//
// Output T_har = (L_high - n_fft + n_fft) / hop + 1 = L_high/hop + 1
//              = 600*T_frames/5 + 1 = 120*T_frames + 1.
// ---------------------------------------------------------------------------

static const float kKokoroTwoPi = 6.283185307179586f;

static float* kokoro_make_har(const float* f0_curve, int T_frames, int* out_T_har, std::mt19937& rng,
                              const float* m_source_w, float m_source_b) {
    const int upsample_scale = 300; // prod(upsample_rates) * hop
    const int harmonic_num = 8;
    const int dim = harmonic_num + 1; // 9
    const int sample_rate = 24000;
    const float sine_amp = 0.1f;
    const float noise_std = 0.003f;
    const float voiced_threshold = 10.0f;
    const int n_fft = 20;
    const int hop = 5;
    const int n_bins = n_fft / 2 + 1; // 11

    const int L_low = 2 * T_frames;
    const int L_high = upsample_scale * L_low;

    // ---- f0_upsamp: nn.Upsample(scale_factor=300) → default mode='nearest'.
    // Each f0 sample is repeated upsample_scale times.
    std::vector<float> f0_up((size_t)L_high);
    for (int i = 0; i < L_low; i++) {
        const float v = f0_curve[i];
        for (int j = 0; j < upsample_scale; j++)
            f0_up[(size_t)i * upsample_scale + j] = v;
    }

    // ---- SineGen._f02sine ----
    // 1. fn[t, k] = f0[t] * (k+1)
    // 2. rad[t, k] = (fn[t, k] / sr) mod 1
    // 3. rand_ini at t=0 for k>=1
    // 4. Downsample rad by 1/upsample_scale (linear interp, align_corners=False)
    // 5. cumsum * 2π → phase_low
    // 6. Upsample phase_low * upsample_scale by upsample_scale (linear)
    // 7. sin(phase) * sine_amp = sines
    std::vector<float> rad((size_t)L_high * dim);
    for (int t = 0; t < L_high; t++) {
        const float fv = f0_up[t];
        for (int k = 0; k < dim; k++) {
            float fn = fv * (float)(k + 1);
            float r = fn / (float)sample_rate;
            r = r - std::floor(r);
            rad[(size_t)t * dim + k] = r;
        }
    }
    // rand_ini: per-(B, dim) noise, B=1; entry at [0] zeroed; only first time step gets noise.
    std::uniform_real_distribution<float> uni01(0.0f, 1.0f);
    std::vector<float> rand_ini((size_t)dim, 0.0f);
    for (int k = 1; k < dim; k++)
        rand_ini[k] = uni01(rng);
    for (int k = 0; k < dim; k++)
        rad[(size_t)0 * dim + k] += rand_ini[k];

    // Downsample rad (L_high → L_low). PyTorch F.interpolate(linear, align_corners=False):
    //   src_idx = (dst + 0.5) * scale - 0.5    (with scale = L_high / L_low = upsample_scale)
    std::vector<float> rad_low((size_t)L_low * dim);
    for (int j = 0; j < L_low; j++) {
        float src_idx = ((float)j + 0.5f) * (float)upsample_scale - 0.5f;
        float clamped = std::max(0.0f, std::min((float)(L_high - 1), src_idx));
        int i0 = (int)std::floor(clamped);
        int i1 = std::min(i0 + 1, L_high - 1);
        float frac = clamped - (float)i0;
        for (int k = 0; k < dim; k++) {
            float v0 = rad[(size_t)i0 * dim + k];
            float v1 = rad[(size_t)i1 * dim + k];
            rad_low[(size_t)j * dim + k] = v0 * (1.0f - frac) + v1 * frac;
        }
    }
    // cumsum * 2π
    std::vector<float> phase_low((size_t)L_low * dim);
    for (int k = 0; k < dim; k++) {
        float acc = 0.0f;
        for (int t = 0; t < L_low; t++) {
            acc += rad_low[(size_t)t * dim + k];
            phase_low[(size_t)t * dim + k] = acc * kKokoroTwoPi;
        }
    }
    // Upsample phase_low * upsample_scale (L_low → L_high), linear interp.
    std::vector<float> sines((size_t)L_high * dim);
    for (int t = 0; t < L_high; t++) {
        float src_idx = ((float)t + 0.5f) / (float)upsample_scale - 0.5f;
        float clamped = std::max(0.0f, std::min((float)(L_low - 1), src_idx));
        int i0 = (int)std::floor(clamped);
        int i1 = std::min(i0 + 1, L_low - 1);
        float frac = clamped - (float)i0;
        for (int k = 0; k < dim; k++) {
            float v0 = phase_low[(size_t)i0 * dim + k] * (float)upsample_scale;
            float v1 = phase_low[(size_t)i1 * dim + k] * (float)upsample_scale;
            float ph = v0 * (1.0f - frac) + v1 * frac;
            sines[(size_t)t * dim + k] = std::sin(ph) * sine_amp;
        }
    }

    // uv mask + noise + apply.
    std::normal_distribution<float> norm(0.0f, 1.0f);
    std::vector<float> sine_waves((size_t)L_high * dim);
    for (int t = 0; t < L_high; t++) {
        const float uv = (f0_up[t] > voiced_threshold) ? 1.0f : 0.0f;
        const float noise_a = uv * noise_std + (1.0f - uv) * sine_amp / 3.0f;
        for (int k = 0; k < dim; k++) {
            float n = noise_a * norm(rng);
            sine_waves[(size_t)t * dim + k] = sines[(size_t)t * dim + k] * uv + n;
        }
    }

    // ---- m_source: l_linear (9 → 1) + tanh (host-cached weights) ----
    std::vector<float> har_source((size_t)L_high);
    for (int t = 0; t < L_high; t++)
        har_source[t] = core_m_source::apply(&sine_waves[(size_t)t * dim], m_source_w, m_source_b);

    // ---- STFT (n_fft=20, hop=5, hann periodic, center=True, pad_mode=reflect) ----
    const int pad_n = n_fft / 2; // 10
    const int L_pad = L_high + 2 * pad_n;
    std::vector<float> padded((size_t)L_pad);
    for (int t = 0; t < L_high; t++)
        padded[t + pad_n] = har_source[t];
    // PyTorch reflect: 'b a | a b c ... y z | z y' actually non-replicated:
    //   padded[pad - 1 - i] = signal[i + 1]   for i in [0, pad-1]
    //   padded[pad + L + i] = signal[L - 2 - i]
    for (int i = 0; i < pad_n; i++) {
        if (i + 1 < L_high)
            padded[pad_n - 1 - i] = har_source[i + 1];
        if (L_high - 2 - i >= 0)
            padded[pad_n + L_high + i] = har_source[L_high - 2 - i];
    }
    // Hann periodic: w[n] = 0.5 - 0.5 cos(2π n / N).
    std::vector<float> hann((size_t)n_fft);
    for (int n = 0; n < n_fft; n++)
        hann[n] = 0.5f - 0.5f * std::cos(kKokoroTwoPi * (float)n / (float)n_fft);

    const int T_har = L_high / hop + 1;
    // Direct DFT (n_fft=20 makes 20² = 400 ops/frame trivial).
    // Pre-compute twiddles cos/sin for k in 0..n_bins-1, n in 0..n_fft-1.
    std::vector<float> tw_cos((size_t)n_bins * n_fft);
    std::vector<float> tw_sin((size_t)n_bins * n_fft);
    for (int k = 0; k < n_bins; k++) {
        for (int n = 0; n < n_fft; n++) {
            float ang = -kKokoroTwoPi * (float)k * (float)n / (float)n_fft;
            tw_cos[(size_t)k * n_fft + n] = std::cos(ang);
            tw_sin[(size_t)k * n_fft + n] = std::sin(ang);
        }
    }

    float* har = (float*)std::malloc((size_t)22 * T_har * sizeof(float));
    if (!har)
        return nullptr;
    for (int frame = 0; frame < T_har; frame++) {
        const int t0 = frame * hop;
        for (int k = 0; k < n_bins; k++) {
            float re = 0.0f, im = 0.0f;
            const float* tc = &tw_cos[(size_t)k * n_fft];
            const float* ts = &tw_sin[(size_t)k * n_fft];
            for (int n = 0; n < n_fft; n++) {
                const float xn = padded[t0 + n] * hann[n];
                re += xn * tc[n];
                im += xn * ts[n];
            }
            const float mag = std::sqrt(re * re + im * im);
            const float ph = std::atan2(im, re);
            // (22, T_har) channel-major: element (c, t) at offset c + t*22.
            har[(size_t)frame * 22 + k] = mag;
            har[(size_t)frame * 22 + n_bins + k] = ph;
        }
    }
    if (out_T_har)
        *out_T_har = T_har;
    return har;
}

// ---------------------------------------------------------------------------
// Generator graph builder.
//
// Inputs (graph-level):
//   "x_in"   ne=(512, 2*T_frames) F32  — dec_decode_3_out
//   "har"    ne=(22,  T_har)      F32  — pre-computed CPU-side
// Style is baked from the voice pack (decoder half).
//
// Outputs:
//   "gen_pre_post_out"  ne=(128, T_har)  — last leaky_relu output, before conv_post
//   "mag"               ne=(11, T_har)   — exp(x[:11]) after conv_post
//   "phase"             ne=(11, T_har)   — sin(x[11:]) after conv_post
// ---------------------------------------------------------------------------

static ggml_cgraph* kokoro_build_graph_generator(kokoro_context* c, int T_frames, int T_har, int L_raw) {
    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), /*no_alloc=*/true};
    ggml_context* ctx0 = ggml_init(ip);
    // Each AdaINResBlock1 has 3 sub-blocks × ~14 ops (adain ×2 + snake ×2 + conv ×2 + add).
    // 8 such blocks (2 noise_res + 6 resblocks). Plus 2 ups, 2 noise_convs, conv_post.
    // Roughly 8*40 + 6*30 + 60 = 600 ops, comfortably within 32k.
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 65536, false);

    ggml_tensor* x = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 512, 2 * T_frames);
    ggml_set_name(x, "x_in");
    ggml_set_input(x);

    ggml_tensor* har = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 22, T_har);
    ggml_set_name(har, "har");
    ggml_set_input(har);

    ggml_tensor* style = kokoro_voice_style_dec_view(ctx0, c, L_raw); // (128, 1)

    // Load all generator weights up-front.
    ggml_tensor* ups0_w = require(c, "dec.gen.ups.0.weight");
    ggml_tensor* ups0_b = require(c, "dec.gen.ups.0.bias");
    ggml_tensor* ups1_w = require(c, "dec.gen.ups.1.weight");
    ggml_tensor* ups1_b = require(c, "dec.gen.ups.1.bias");
    ggml_tensor* nc0_w = require(c, "dec.gen.noise_convs.0.weight");
    ggml_tensor* nc0_b = require(c, "dec.gen.noise_convs.0.bias");
    ggml_tensor* nc1_w = require(c, "dec.gen.noise_convs.1.weight");
    ggml_tensor* nc1_b = require(c, "dec.gen.noise_convs.1.bias");
    ggml_tensor* cp_w = require(c, "dec.gen.conv_post.weight");
    ggml_tensor* cp_b = require(c, "dec.gen.conv_post.bias");

    const int dilations[3] = {1, 3, 5};
    kokoro_resblock1_w noise_res0, noise_res1;
    kokoro_load_resblock1(c, noise_res0, "dec.gen.noise_res.0", /*K=*/7, dilations);
    kokoro_load_resblock1(c, noise_res1, "dec.gen.noise_res.1", /*K=*/11, dilations);
    kokoro_resblock1_w resblocks[6];
    const int rb_K[3] = {3, 7, 11};
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            char prefix[64];
            std::snprintf(prefix, sizeof(prefix), "dec.gen.resblocks.%d", i * 3 + j);
            kokoro_load_resblock1(c, resblocks[i * 3 + j], prefix, rb_K[j], dilations);
        }
    }

    const bool time_first = kokoro_generator_time_first(c);

    // Prepared Snake-α operands for the Vulkan (T, C) path. Expanded into `gf`
    // before any Snake chain so reciprocal DIVs cannot break MUL→SIN→SQR→MUL→ADD.
    kokoro_snake_alpha_tf noise_a1[2][3] = {};
    kokoro_snake_alpha_tf noise_a2[2][3] = {};
    kokoro_snake_alpha_tf res_a1[6][3] = {};
    kokoro_snake_alpha_tf res_a2[6][3] = {};
    if (time_first) {
        // Host uploads remain channel-first (C, T); convert once at the boundary.
        x = ggml_cont(ctx0, ggml_transpose(ctx0, x));     // (2*T_frames, 512)
        har = ggml_cont(ctx0, ggml_transpose(ctx0, har)); // (T_har, 22)

        auto prep_block = [&](const kokoro_resblock1_w& w, kokoro_snake_alpha_tf a1[3], kokoro_snake_alpha_tf a2[3]) {
            for (int j = 0; j < 3; j++) {
                a1[j] = kokoro_snake_alpha_prepare_tf(ctx0, w.alpha1[j]);
                a2[j] = kokoro_snake_alpha_prepare_tf(ctx0, w.alpha2[j]);
                ggml_build_forward_expand(gf, a1[j].inv_a);
                ggml_build_forward_expand(gf, a2[j].inv_a);
            }
        };
        prep_block(noise_res0, noise_a1[0], noise_a2[0]);
        prep_block(noise_res1, noise_a1[1], noise_a2[1]);
        for (int b = 0; b < 6; b++)
            prep_block(resblocks[b], res_a1[b], res_a2[b]);
    }

    // ---- Upsample loop ----
    for (int i = 0; i < 2; i++) {
        // x = LeakyReLU(0.1)(x)
        x = ggml_leaky_relu(ctx0, x, /*slope=*/0.1f, /*inplace=*/false);

        // x_source = noise_convs[i](har); noise_res[i](x_source, s_dec)
        ggml_tensor* x_source;
        if (time_first) {
            if (i == 0) {
                x_source = kokoro_conv1d_ks_tf(ctx0, har, nc0_w, nc0_b, /*s*/ 6, /*p*/ 3);
                x_source = kokoro_resblock1_forward_tf(ctx0, x_source, style, noise_res0, noise_a1[0], noise_a2[0]);
            } else {
                x_source = kokoro_conv1d_ks_tf(ctx0, har, nc1_w, nc1_b, /*s*/ 1, /*p*/ 0);
                x_source = kokoro_resblock1_forward_tf(ctx0, x_source, style, noise_res1, noise_a1[1], noise_a2[1]);
            }
        } else if (i == 0) {
            // k=12, s=6, p=(s+1)/2=3
            x_source = kokoro_conv1d_ks(ctx0, har, nc0_w, nc0_b, /*s*/ 6, /*p*/ 3);
            x_source = kokoro_resblock1_forward(ctx0, x_source, style, noise_res0);
        } else {
            // k=1, s=1, p=0
            x_source = kokoro_conv1d_ks(ctx0, har, nc1_w, nc1_b, /*s*/ 1, /*p*/ 0);
            x_source = kokoro_resblock1_forward(ctx0, x_source, style, noise_res1);
        }

        // x = ups[i](x)  with PyTorch padding handled via post-crop
        if (time_first) {
            if (i == 0) {
                x = kokoro_convt1d_pad_tf(ctx0, x, ups0_w, c->ups_w_perm[0], ups0_b, /*s*/ 10, /*p*/ 5);
            } else {
                x = kokoro_convt1d_pad_tf(ctx0, x, ups1_w, c->ups_w_perm[1], ups1_b, /*s*/ 6, /*p*/ 3);
            }
        } else if (i == 0) {
            // k=20, s=10, p=(k-s)/2 = 5
            x = kokoro_convt1d_pad(ctx0, x, ups0_w, c->ups_w_perm[0], ups0_b, /*s*/ 10, /*p*/ 5);
        } else {
            // k=12, s=6, p=(k-s)/2 = 3
            x = kokoro_convt1d_pad(ctx0, x, ups1_w, c->ups_w_perm[1], ups1_b, /*s*/ 6, /*p*/ 3);
        }

        // Last upsample: reflection_pad((1, 0)) — 1 sample on the left, 0 on the right.
        // ggml_pad_reflect_1d pads ne[0]; time-first is already (T, C).
        if (i == 1) {
            if (time_first) {
                x = ggml_pad_reflect_1d(ctx0, x, /*left=*/1, /*right=*/0); // (T+1, C)
            } else {
                ggml_tensor* xT = ggml_cont(ctx0, ggml_transpose(ctx0, x));   // (T, C)
                xT = ggml_pad_reflect_1d(ctx0, xT, /*left=*/1, /*right=*/0); // (T+1, C)
                x = ggml_cont(ctx0, ggml_transpose(ctx0, xT));               // (C, T+1)
            }
        }

        x = ggml_add(ctx0, x, x_source);

        // Average of 3 resblocks at indices i*3+0, i*3+1, i*3+2.
        ggml_tensor* xs = nullptr;
        for (int j = 0; j < 3; j++) {
            ggml_tensor* xj;
            if (time_first) {
                const int bi = i * 3 + j;
                xj = kokoro_resblock1_forward_tf(ctx0, x, style, resblocks[bi], res_a1[bi], res_a2[bi]);
            } else {
                xj = kokoro_resblock1_forward(ctx0, x, style, resblocks[i * 3 + j]);
            }
            xs = (xs == nullptr) ? xj : ggml_add(ctx0, xs, xj);
        }
        x = ggml_scale(ctx0, xs, 1.0f / 3.0f);
    }

    // x = LeakyReLU(0.01)(x)  — default PyTorch slope, NOT 0.1.
    x = ggml_leaky_relu(ctx0, x, /*slope=*/0.01f, /*inplace=*/false);

    ggml_tensor* gen_pre_post;
    if (time_first) {
        // Named debug/output tensor stays channel-first (128, T_har) for layout
        // compatibility; conv_post still consumes the live (T, C) activation.
        gen_pre_post = ggml_cont(ctx0, ggml_transpose(ctx0, x)); // (128, T_har)
    } else {
        gen_pre_post = ggml_cont(ctx0, x);
    }
    ggml_set_name(gen_pre_post, "gen_pre_post_out");
    ggml_set_output(gen_pre_post);
    ggml_build_forward_expand(gf, gen_pre_post);

    // conv_post: Conv1d(128, 22, k=7, p=3). Same-padding output length = T_har.
    if (time_first) {
        x = kokoro_conv1d_ks_tf(ctx0, x, cp_w, cp_b, /*s*/ 1, /*p*/ 3); // (T_har, 22)
        x = ggml_cont(ctx0, ggml_transpose(ctx0, x));                   // (22, T_har)
    } else {
        x = kokoro_conv1d_ks(ctx0, gen_pre_post, cp_w, cp_b, /*s*/ 1, /*p*/ 3); // (22, T_har)
    }

    // Split (22, T_har) along ne[0]: rows 0..10 = mag-side, 11..21 = phase-side.
    // View along ne[0] is non-contiguous (stride between rows mismatches ne[0]),
    // so cont after the view to give exp/sin a clean buffer.
    ggml_tensor* mag_view = ggml_view_2d(ctx0, x, 11, T_har, x->nb[1], 0);
    ggml_tensor* mag_in = ggml_cont(ctx0, mag_view);
    ggml_tensor* mag = ggml_exp(ctx0, mag_in);
    ggml_set_name(mag, "mag");
    ggml_set_output(mag);
    ggml_build_forward_expand(gf, mag);

    ggml_tensor* phase_view = ggml_view_2d(ctx0, x, 11, T_har, x->nb[1], (size_t)11 * sizeof(float));
    ggml_tensor* phase_in = ggml_cont(ctx0, phase_view);
    ggml_tensor* phase = ggml_sin(ctx0, phase_in);
    ggml_set_name(phase, "phase");
    ggml_set_output(phase);
    ggml_build_forward_expand(gf, phase);

    ggml_free(ctx0);
    return gf;
}


// ---------------------------------------------------------------------------
// M8 — iSTFT (CPU)
//
// Reference: torch.istft(filter_length=20, hop=5, win_length=20,
//                        window=hann_window(20, periodic=True), center=True).
//
// For each frame m in 0..T_har-1:
//   X[k] = mag[k, m] * exp(j * phase[k, m])    for k in 0..10  (n_bins)
//   Mirror to full 20-bin: X[20-k] = conj(X[k]) for k in 1..9
//   y_frame[n] = (1/N) * Re(sum_k X[k] * exp(j*2π*k*n/N))      for n in 0..19
//   y_frame *= window
//   out[m*hop : m*hop + N] += y_frame
//   ola_norm[m*hop : m*hop + N] += window²
// out /= ola_norm  (where ola_norm > 0); strip pad samples.
// Output length = (T_har - 1) * hop = 600 * T_frames.
// ---------------------------------------------------------------------------

static float* kokoro_run_istft(const float* mag, const float* phase, int T_har, int* out_T_audio) {
    kokoro_bench_stage _b("istft");
    const int n_fft = 20;
    const int hop = 5;
    const int n_bins = n_fft / 2 + 1;
    const int pad_n = n_fft / 2;

    // Hann periodic.
    std::vector<float> hann((size_t)n_fft);
    for (int n = 0; n < n_fft; n++)
        hann[n] = 0.5f - 0.5f * std::cos(kKokoroTwoPi * (float)n / (float)n_fft);

    // L_padded covers the OLA region; L_audio is the unpadded output (center=True
    // strips n_fft/2 samples from each end).
    const int L_padded = (T_har - 1) * hop + n_fft;
    const int L_audio = L_padded - 2 * pad_n;
    if (L_audio <= 0) {
        if (out_T_audio)
            *out_T_audio = 0;
        return nullptr;
    }

    std::vector<float> y((size_t)L_padded, 0.0f);
    std::vector<float> wsum((size_t)L_padded, 0.0f);

    // Pre-compute IDFT twiddles: cos/sin(2π k n / N) for k in 0..n_bins-1.
    // For k in [n_bins, n_fft-1] we exploit Hermitian symmetry (see below).
    std::vector<float> tw_cos((size_t)n_bins * n_fft);
    std::vector<float> tw_sin((size_t)n_bins * n_fft);
    for (int k = 0; k < n_bins; k++) {
        for (int n = 0; n < n_fft; n++) {
            float ang = kKokoroTwoPi * (float)k * (float)n / (float)n_fft;
            tw_cos[(size_t)k * n_fft + n] = std::cos(ang);
            tw_sin[(size_t)k * n_fft + n] = std::sin(ang);
        }
    }

    // Per frame: IDFT directly from half-spectrum using Hermitian symmetry.
    // For real-valued y[n]:
    //   y[n] = (1/N) * (Xr[0] + (-1)^n * Xr[N/2]
    //                   + 2 * sum_{k=1}^{N/2-1} (Xr[k]*cos(2πkn/N) - Xi[k]*sin(2πkn/N)))
    const float invN = 1.0f / (float)n_fft;
    for (int m = 0; m < T_har; m++) {
        // mag/phase have layout (n_bins, T_har) F32 contiguous: element (k, m) at k + m*n_bins.
        const int t0 = m * hop;
        for (int n = 0; n < n_fft; n++) {
            float acc = mag[(size_t)m * n_bins + 0]; // X[0] (real)
            // X[N/2]: real, sign alternates with n.
            const float xN2 =
                mag[(size_t)m * n_bins + (n_bins - 1)] * std::cos(phase[(size_t)m * n_bins + (n_bins - 1)]);
            acc += ((n & 1) ? -xN2 : xN2);
            // k = 1..N/2-1: doubled Hermitian pair.
            for (int k = 1; k < n_bins - 1; k++) {
                const float r = mag[(size_t)m * n_bins + k];
                const float ph = phase[(size_t)m * n_bins + k];
                const float xr = r * std::cos(ph);
                const float xi = r * std::sin(ph);
                acc += 2.0f * (xr * tw_cos[(size_t)k * n_fft + n] - xi * tw_sin[(size_t)k * n_fft + n]);
            }
            const float yval = acc * invN * hann[n];
            y[(size_t)t0 + n] += yval;
            wsum[(size_t)t0 + n] += hann[n] * hann[n];
        }
    }

    // Normalize by window-sum and strip the center=True padding.
    float* out = (float*)std::malloc((size_t)L_audio * sizeof(float));
    if (!out) {
        if (out_T_audio)
            *out_T_audio = 0;
        return nullptr;
    }
    const float wsum_eps = 1e-11f;
    for (int t = 0; t < L_audio; t++) {
        const float w = wsum[(size_t)pad_n + t];
        out[t] = (w > wsum_eps) ? y[(size_t)pad_n + t] / w : 0.0f;
    }
    if (out_T_audio)
        *out_T_audio = L_audio;
    return out;
}


// ---------------------------------------------------------------------------
// Unified synthesis pipeline — runs each graph at most once per call.
//
// The selector-style kokoro_run_* helpers above rerun earlier stages for
// each output they need. For production synthesis, this pipeline executes
// each expensive graph exactly once and passes intermediates forward.
// ---------------------------------------------------------------------------

static float* kokoro_synthesize_pipeline(kokoro_context* c, const int32_t* raw_ids, int n_raw, int* out_n) {
    const auto pipeline_start = kokoro_bench_clock::now();
    if (out_n) *out_n = 0;
    if (!c->vp_loaded || n_raw <= 0) return nullptr;

    // --- Stage 1: BERT + duration predictor (one functional graph) ---
    // bert_proj_out stays device-resident and feeds the predictor directly,
    // removing the prior D2H host vector, H2D upload, and extra sched reset.
    auto stage_start = kokoro_bench_clock::now();
    std::vector<int32_t> padded;
    padded.reserve((size_t)n_raw + 2);
    padded.push_back(c->vocab.pad_id);
    padded.insert(padded.end(), raw_ids, raw_ids + n_raw);
    padded.push_back(c->vocab.pad_id);
    const int L = (int)padded.size();
    if (L > (int)c->hp.plbert_max_pos) {
        kokoro_set_error("padded phoneme count %d exceeds model maximum %u", L, c->hp.plbert_max_pos);
        return nullptr;
    }

    ggml_cgraph* gf_bp = kokoro_build_graph_bert_predictor(c, L, n_raw);
    if (!gf_bp) return nullptr;
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf_bp)) return nullptr;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf_bp, "ids"), padded.data(), 0, (size_t)L * sizeof(int32_t));
    std::vector<int32_t> positions(L);
    for (int i = 0; i < L; i++) positions[i] = i;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf_bp, "positions"), positions.data(), 0, (size_t)L * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(c->sched, gf_bp) != GGML_STATUS_SUCCESS) return nullptr;

    // Extract dur_enc_out (640, L)
    ggml_tensor* de_t = ggml_graph_get_tensor(gf_bp, "dur_enc_out");
    const size_t de_n = (size_t)de_t->ne[0] * (size_t)de_t->ne[1];
    std::vector<float> dur_enc(de_n);
    ggml_backend_tensor_get(de_t, dur_enc.data(), 0, de_n * sizeof(float));

    // Extract durations (1, L) → round + clamp after the combined stage timer.
    ggml_tensor* dp_t = ggml_graph_get_tensor(gf_bp, "durations_pre");
    std::vector<float> dur_raw((size_t)L);
    ggml_backend_tensor_get(dp_t, dur_raw.data(), 0, (size_t)L * sizeof(float));
    kokoro_bench_elapsed("bert_predictor", stage_start);

    stage_start = kokoro_bench_clock::now();
    const float length_scale = c->params.length_scale > 0.0f ? c->params.length_scale : 1.0f;
    const bool debug_durations = kokoro_env::truthy("KOKORO_DEBUG_DURATIONS");
    std::vector<int> dur_int((size_t)L);
    for (int i = 0; i < L; i++) {
        const float scaled = dur_raw[i] * length_scale;
        float rounded = std::nearbyintf(scaled);
        if (rounded < 1.0f)
            rounded = 1.0f;
        dur_int[i] = (int)rounded;
        if (debug_durations) {
            kokoro_log(KOKORO_LOG_DEBUG,
                       "kokoro: duration[%d] raw=%.9g scaled=%.9g rounded=%d",
                       i, (double)dur_raw[i], (double)scaled, dur_int[i]);
        }
    }

    // --- Stage 2: host duration rounding → compact I32 alignment indices ---
    // T_frames is data-dependent; only the small duration vector must sync.
    // Index length must equal T_frames for both later gathers.
    int T_frames = 0;
    int32_t* align_idx = core_align::build_indices(L, dur_int.data(), &T_frames);
    if (!align_idx || T_frames <= 0) { std::free(align_idx); return nullptr; }
    kokoro_bench_elapsed("duration_indices", stage_start);

    // --- Stage 3: F0/N prediction (gather dur_enc on device, then F0/N) ---
    stage_start = kokoro_bench_clock::now();
    ggml_cgraph* gf_f0n = kokoro_build_graph_f0n(c, T_frames, L, n_raw);
    if (!gf_f0n) { std::free(align_idx); return nullptr; }
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf_f0n)) { std::free(align_idx); return nullptr; }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf_f0n, "dur_enc"), dur_enc.data(), 0, de_n * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf_f0n, "align_idx"), align_idx, 0, (size_t)T_frames * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(c->sched, gf_f0n) != GGML_STATUS_SUCCESS) {
        std::free(align_idx);
        return nullptr;
    }

    // Extract both f0 and n curves
    ggml_tensor* f0_t = ggml_graph_get_tensor(gf_f0n, "f0_curve");
    ggml_tensor* n_t = ggml_graph_get_tensor(gf_f0n, "n_curve");
    const size_t fn_n = (size_t)f0_t->ne[0] * (size_t)f0_t->ne[1]; // 2*T_frames
    std::vector<float> f0_curve(fn_n), n_curve(fn_n);
    ggml_backend_tensor_get(f0_t, f0_curve.data(), 0, fn_n * sizeof(float));
    ggml_backend_tensor_get(n_t, n_curve.data(), 0, fn_n * sizeof(float));
    kokoro_bench_elapsed("f0_noise", stage_start);

    // Harmonic source depends only on f0, T_frames, cached m_source, and RNG.
    // On Vulkan, overlap it with TextEncoder + Decoder GPU work. CPU/Metal/CUDA
    // keep the historical serial order. KOKORO_HAR_OVERLAP=0 disables overlap.
    const char* seed_env = kokoro_env::get("KOKORO_SEED");
    const uint32_t seed = seed_env ? (uint32_t)std::strtoul(seed_env, nullptr, 0) : 0x12345u;
    std::mt19937 rng(seed);
    bool overlap_har = (c->selected_backend == KOKORO_BACKEND_VULKAN) &&
                       (kokoro_env::get("KOKORO_HAR_OVERLAP") == nullptr ||
                        kokoro_env::truthy("KOKORO_HAR_OVERLAP"));
    if (!c->m_source_cached) {
        std::free(align_idx);
        kokoro_set_error("m_source host cache missing");
        return nullptr;
    }

    // Declared after f0_curve so destruction joins while f0 data is still alive.
    core_har::AsyncResult har_async;
    if (overlap_har) {
        std::array<float, core_m_source::kDim> mw{};
        std::memcpy(mw.data(), c->m_source_weight, sizeof(mw));
        const float mb = c->m_source_bias;
        const float* f0_ptr = f0_curve.data(); // immutable until har_async joins
        const int T_frames_cap = T_frames;
        try {
            har_async = core_har::AsyncResult(std::async(
                std::launch::async,
                [f0_ptr, T_frames_cap, mw, mb, rng = std::move(rng)]() mutable -> core_har::ComputeResult {
                    core_har::ComputeResult r;
                    const auto t0 = kokoro_bench_clock::now();
                    r.data = kokoro_make_har(f0_ptr, T_frames_cap, &r.T_har, rng, mw.data(), mb);
                    r.work_ms = std::chrono::duration<double, std::milli>(kokoro_bench_clock::now() - t0).count();
                    return r;
                }));
        } catch (...) {
            // Thread creation can fail under resource pressure. Preserve
            // synthesis correctness by falling back to the serial path.
            overlap_har = false;
            rng.seed(seed);
        }
    }

    // --- Stage 4: TextEncoder ---
    stage_start = kokoro_bench_clock::now();
    ggml_cgraph* gf_te = kokoro_build_graph_text_enc(c, L);
    if (!gf_te) { std::free(align_idx); return nullptr; }
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf_te)) { std::free(align_idx); return nullptr; }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf_te, "ids"), padded.data(), 0, (size_t)L * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(c->sched, gf_te) != GGML_STATUS_SUCCESS) {
        std::free(align_idx);
        return nullptr;
    }
    ggml_tensor* te_t = ggml_graph_get_tensor(gf_te, "text_enc_out");
    const size_t te_n = (size_t)te_t->ne[0] * (size_t)te_t->ne[1];
    std::vector<float> text_enc(te_n);
    ggml_backend_tensor_get(te_t, text_enc.data(), 0, te_n * sizeof(float));
    kokoro_bench_elapsed("text_encoder", stage_start);

    // --- Stage 5: Decoder body (gather text_enc on device with same indices) ---
    stage_start = kokoro_bench_clock::now();
    ggml_cgraph* gf_dec = kokoro_build_graph_decoder_body(c, T_frames, L, n_raw);
    if (!gf_dec) { std::free(align_idx); return nullptr; }
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf_dec)) { std::free(align_idx); return nullptr; }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf_dec, "text_enc"), text_enc.data(), 0, te_n * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf_dec, "align_idx"), align_idx, 0, (size_t)T_frames * sizeof(int32_t));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf_dec, "f0"), f0_curve.data(), 0, fn_n * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf_dec, "n"), n_curve.data(), 0, fn_n * sizeof(float));
    std::free(align_idx);
    if (ggml_backend_sched_graph_compute(c->sched, gf_dec) != GGML_STATUS_SUCCESS) return nullptr;

    ggml_tensor* dec_out = ggml_graph_get_tensor(gf_dec, "dec_decode_3_out");
    const size_t dec_n = (size_t)dec_out->ne[0] * (size_t)dec_out->ne[1];
    std::vector<float> dec_decode(dec_n);
    ggml_backend_tensor_get(dec_out, dec_decode.data(), 0, dec_n * sizeof(float));
    kokoro_bench_elapsed("decoder", stage_start);

    // --- Stage 6: HnNSF (CPU-side har generation; possibly already running) ---
    float* har = nullptr;
    int T_har = 0;
    if (overlap_har) {
        const auto wait_start = kokoro_bench_clock::now();
        try {
            auto& r = har_async.get();
            kokoro_bench_elapsed("harmonic_wait", wait_start);
            // Actual worker time, logged on the caller thread (not from the worker).
            kokoro_bench_report("harmonic_source", r.work_ms);
            if (!r.data)
                return nullptr;
            T_har = r.T_har;
            har = har_async.release();
        } catch (const std::exception& e) {
            kokoro_set_error("harmonic generation failed: %s", e.what());
            return nullptr;
        } catch (...) {
            kokoro_set_error("harmonic generation failed");
            return nullptr;
        }
    } else {
        stage_start = kokoro_bench_clock::now();
        har = kokoro_make_har(f0_curve.data(), T_frames, &T_har, rng, c->m_source_weight, c->m_source_bias);
        if (!har)
            return nullptr;
        kokoro_bench_elapsed("harmonic_source", stage_start);
    }

    // --- Stage 7: Generator (produces mag + phase in one graph) ---
    stage_start = kokoro_bench_clock::now();
    ggml_cgraph* gf_gen = kokoro_build_graph_generator(c, T_frames, T_har, n_raw);
    if (!gf_gen) { std::free(har); return nullptr; }
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf_gen)) { std::free(har); return nullptr; }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf_gen, "x_in"), dec_decode.data(), 0, dec_n * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf_gen, "har"), har, 0, (size_t)22 * T_har * sizeof(float));
    std::free(har);
    if (ggml_backend_sched_graph_compute(c->sched, gf_gen) != GGML_STATUS_SUCCESS) return nullptr;

    // Extract both mag and phase from one generator execution
    ggml_tensor* mag_t = ggml_graph_get_tensor(gf_gen, "mag");
    ggml_tensor* phase_t = ggml_graph_get_tensor(gf_gen, "phase");
    const size_t mp_n = (size_t)mag_t->ne[0] * (size_t)mag_t->ne[1]; // 11 * T_har
    std::vector<float> mag(mp_n), phase(mp_n);
    ggml_backend_tensor_get(mag_t, mag.data(), 0, mp_n * sizeof(float));
    ggml_backend_tensor_get(phase_t, phase.data(), 0, mp_n * sizeof(float));
    kokoro_bench_elapsed("generator", stage_start);

    // --- Stage 8: iSTFT (CPU) ---
    int T_audio = 0;
    float* audio = kokoro_run_istft(mag.data(), phase.data(), T_har, &T_audio);
    if (!audio) return nullptr;
    if (out_n) *out_n = T_audio;
    kokoro_bench_elapsed("pipeline_total", pipeline_start);
    return audio;
}

} // namespace

// ---------------------------------------------------------------------------
// C ABI
// ---------------------------------------------------------------------------

extern "C" struct kokoro_context_params kokoro_context_default_params(void) {
    kokoro_context_params p{};
    p.abi_version = KOKORO_ABI_VERSION;
    p.n_threads = 4;
    p.verbosity = 1;
    p.length_scale = 1.0f;
    p.backend = KOKORO_BACKEND_AUTO;
    p.device = 0;
    std::strncpy(p.espeak_lang, "en-us", sizeof(p.espeak_lang) - 1);
    return p;
}

extern "C" struct kokoro_context* kokoro_init_from_file(const char* path_model, struct kokoro_context_params params) {
    if (!path_model) {
        kokoro_set_error("null model path");
        return nullptr;
    }
    if (params.abi_version != KOKORO_ABI_VERSION) {
        kokoro_set_error("unsupported abi_version %d (library requires %d)",
                        params.abi_version, KOKORO_ABI_VERSION);
        return nullptr;
    }
    if (params.backend < KOKORO_BACKEND_AUTO || params.backend > KOKORO_BACKEND_CUDA) {
        kokoro_set_error("invalid backend value %d", (int)params.backend);
        return nullptr;
    }
    if (params.device < 0) {
        kokoro_set_error("device must be >= 0");
        return nullptr;
    }
    // Match PyTorch's torch.round behaviour (banker's rounding) when the
    // runtime later applies `nearbyintf` to durations (R5 in the plan).
    // Setting this globally is benign — the only place we round is the
    // duration post-processing.
    std::fesetround(FE_TONEAREST);
    auto* c = new kokoro_context();
    c->params = params;
    c->n_threads = params.n_threads > 0 ? params.n_threads : 4;
    if (params.espeak_lang[0])
        c->espeak_lang = params.espeak_lang;

    // ---- Pass 1: metadata (hparams + vocab) ----
    {
        gguf_context* g = core_gguf::open_metadata(path_model);
        if (!g) {
            kokoro_set_error("failed to read GGUF '%s'", path_model);
            delete c;
            return nullptr;
        }

        // Architecture sanity. The converter writes "kokoro" for both
        // hexgrad/Kokoro-82M and yl4579/StyleTTS2-LJSpeech.
        std::string arch = core_gguf::kv_str(g, "general.architecture", "");
        if (arch != "kokoro") {
            kokoro_set_error("unexpected general.architecture='%s' (want 'kokoro')", arch.c_str());
            gguf_free(g);
            delete c;
            return nullptr;
        }

        auto& hp = c->hp;
        hp.hidden_dim = core_gguf::kv_u32(g, "kokoro.hidden_dim", hp.hidden_dim);
        hp.style_dim = core_gguf::kv_u32(g, "kokoro.style_dim", hp.style_dim);
        hp.max_dur = core_gguf::kv_u32(g, "kokoro.max_dur", hp.max_dur);
        hp.n_token = core_gguf::kv_u32(g, "kokoro.n_token", hp.n_token);
        hp.n_mels = core_gguf::kv_u32(g, "kokoro.n_mels", hp.n_mels);
        hp.n_layer = core_gguf::kv_u32(g, "kokoro.n_layer", hp.n_layer);
        hp.text_enc_k = core_gguf::kv_u32(g, "kokoro.text_encoder_kernel_size", hp.text_enc_k);
        hp.sample_rate = core_gguf::kv_u32(g, "kokoro.sample_rate", hp.sample_rate);
        hp.vocab_size = core_gguf::kv_u32(g, "kokoro.vocab_size", hp.vocab_size);

        hp.plbert_embd_size = core_gguf::kv_u32(g, "kokoro.plbert.embedding_size", hp.plbert_embd_size);
        hp.plbert_hidden = core_gguf::kv_u32(g, "kokoro.plbert.hidden_size", hp.plbert_hidden);
        hp.plbert_n_layers = core_gguf::kv_u32(g, "kokoro.plbert.num_hidden_layers", hp.plbert_n_layers);
        hp.plbert_n_heads = core_gguf::kv_u32(g, "kokoro.plbert.num_attention_heads", hp.plbert_n_heads);
        hp.plbert_ff = core_gguf::kv_u32(g, "kokoro.plbert.intermediate_size", hp.plbert_ff);
        hp.plbert_max_pos = core_gguf::kv_u32(g, "kokoro.plbert.max_position_embeddings", hp.plbert_max_pos);
        hp.plbert_vocab_size = core_gguf::kv_u32(g, "kokoro.plbert.vocab_size", hp.plbert_vocab_size);

        hp.istft_init_ch = core_gguf::kv_u32(g, "kokoro.istft.init_channel", hp.istft_init_ch);
        hp.istft_n_fft = core_gguf::kv_u32(g, "kokoro.istft.n_fft", hp.istft_n_fft);
        hp.istft_hop = core_gguf::kv_u32(g, "kokoro.istft.hop_size", hp.istft_hop);
        hp.istft_n_dilations = core_gguf::kv_u32(g, "kokoro.istft.resblock_n_dilations", hp.istft_n_dilations);

        hp.istft_upsample_rates = kv_u32_array(g, "kokoro.istft.upsample_rates");
        hp.istft_upsample_kernel_sizes = kv_u32_array(g, "kokoro.istft.upsample_kernel_sizes");
        hp.istft_resblock_kernel_sizes = kv_u32_array(g, "kokoro.istft.resblock_kernel_sizes");
        hp.istft_resblock_dilations = kv_u32_array(g, "kokoro.istft.resblock_dilation_sizes");

        if (hp.istft_upsample_rates.empty())
            hp.istft_upsample_rates = {10, 6};
        if (hp.istft_upsample_kernel_sizes.empty())
            hp.istft_upsample_kernel_sizes = {20, 12};
        if (hp.istft_resblock_kernel_sizes.empty())
            hp.istft_resblock_kernel_sizes = {3, 7, 11};
        if (hp.istft_resblock_dilations.empty())
            hp.istft_resblock_dilations = {1, 3, 5, 1, 3, 5, 1, 3, 5};

        c->vocab.id_to_token = core_gguf::kv_str_array(g, "tokenizer.ggml.tokens");
        c->vocab.token_to_id.reserve(c->vocab.id_to_token.size());
        for (int i = 0; i < (int)c->vocab.id_to_token.size(); i++) {
            const auto& t = c->vocab.id_to_token[i];
            if (!t.empty())
                c->vocab.token_to_id[t] = i;
        }
        // StyleTTS2/Kokoro convention: index 0 is the "$" pad token.
        c->vocab.pad_id = 0;

        gguf_free(g);
    }

    if (params.verbosity >= 1) {
        const auto& hp = c->hp;
        kokoro_log(KOKORO_LOG_INFO,
                "kokoro: arch=kokoro  hidden=%u style=%u max_dur=%u n_token=%u "
                "vocab=%zu plbert=%uL/%uH/ff=%u  iSTFT n_fft=%u hop=%u sr=%u",
                hp.hidden_dim, hp.style_dim, hp.max_dur, hp.n_token, c->vocab.id_to_token.size(), hp.plbert_n_layers,
                hp.plbert_hidden, hp.plbert_ff, hp.istft_n_fft, hp.istft_hop, hp.sample_rate);
    }

    // ---- Backends ----
    c->backend_cpu = ggml_backend_cpu_init();
    if (!c->backend_cpu) {
        kokoro_set_error("failed to init CPU backend");
        delete c;
        return nullptr;
    }
    ggml_backend_cpu_set_n_threads(c->backend_cpu, c->n_threads);

    const enum kokoro_backend requested = params.backend;
    auto init_vulkan = [&]() -> bool {
#ifdef KOKORO_HAS_VULKAN
        if (params.device >= ggml_backend_vk_get_device_count())
            return false;
        c->backend = ggml_backend_vk_init((size_t)params.device);
        if (c->backend)
            c->selected_backend = KOKORO_BACKEND_VULKAN;
        return c->backend != nullptr;
#else
        return false;
#endif
    };
    auto init_metal = [&]() -> bool {
#ifdef KOKORO_HAS_METAL
        if (params.device != 0)
            return false;
        c->backend = ggml_backend_metal_init();
        if (c->backend)
            c->selected_backend = KOKORO_BACKEND_METAL;
        return c->backend != nullptr;
#else
        return false;
#endif
    };
    auto init_cuda = [&]() -> bool {
#ifdef KOKORO_HAS_CUDA
        if (params.device >= ggml_backend_cuda_get_device_count())
            return false;
        c->backend = ggml_backend_cuda_init(params.device);
        if (c->backend)
            c->selected_backend = KOKORO_BACKEND_CUDA;
        return c->backend != nullptr;
#else
        return false;
#endif
    };

    switch (requested) {
    case KOKORO_BACKEND_AUTO:
        if (!init_metal() && !init_cuda())
            init_vulkan();
        break;
    case KOKORO_BACKEND_CPU:
        break;
    case KOKORO_BACKEND_VULKAN:
        if (!init_vulkan()) {
            kokoro_set_error("Vulkan backend unavailable or device %d is invalid", params.device);
            kokoro_free(c);
            return nullptr;
        }
        break;
    case KOKORO_BACKEND_METAL:
        if (!init_metal()) {
            kokoro_set_error("Metal backend unavailable (Metal only supports device 0)");
            kokoro_free(c);
            return nullptr;
        }
        break;
    case KOKORO_BACKEND_CUDA:
        if (!init_cuda()) {
            kokoro_set_error("CUDA backend unavailable or device %d is invalid", params.device);
            kokoro_free(c);
            return nullptr;
        }
        break;
    }

    if (!c->backend) {
        c->backend = c->backend_cpu;
        c->selected_backend = KOKORO_BACKEND_CPU;
    }

    if (params.verbosity >= 1)
        kokoro_log(KOKORO_LOG_INFO, "kokoro: backend=%s device=%d",
                   ggml_backend_name(c->backend), params.device);

    // ---- Pass 2: weights ----
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path_model, c->backend, "kokoro", wl)) {
        kokoro_set_error("failed to load weights from '%s'", path_model);
        kokoro_free(c);
        return nullptr;
    }
    c->ctx_w = wl.ctx;
    c->buf_w = wl.buf;
    c->tensors = std::move(wl.tensors);

    if (!sanity_check_weights(c)) {
        kokoro_set_error("weight sanity check failed for '%s'", path_model);
        kokoro_free(c);
        return nullptr;
    }

    if (!kokoro_cache_m_source(c)) {
        // kokoro_cache_m_source already set the error.
        kokoro_free(c);
        return nullptr;
    }

    // ---- Permute ConvTranspose1d weights for decomposed mul_mat + col2im ----
    {
        const char* ups_names[2] = {"dec.gen.ups.0.weight", "dec.gen.ups.1.weight"};
        const size_t meta_bytes = ggml_tensor_overhead() * 2 + 4096;
        struct ggml_init_params pp = {meta_bytes, nullptr, true};
        c->ctx_perm = ggml_init(pp);
        std::unique_ptr<float[]> perm_bufs[2];
        for (int i = 0; i < 2; i++) {
            auto it = c->tensors.find(ups_names[i]);
            if (it == c->tensors.end())
                continue;
            ggml_tensor* src = it->second;
            perm_bufs[i] = core_convt::permute_convt1d_weight(src);
            c->ups_w_perm[i] =
                ggml_new_tensor_2d(c->ctx_perm, GGML_TYPE_F32, (int)src->ne[2], (int)src->ne[0] * (int)src->ne[1]);
        }
        c->buf_perm = ggml_backend_alloc_ctx_tensors(c->ctx_perm, c->backend);
        for (int i = 0; i < 2; i++) {
            if (c->ups_w_perm[i] && perm_bufs[i])
                ggml_backend_tensor_set(c->ups_w_perm[i], perm_bufs[i].get(), 0, ggml_nbytes(c->ups_w_perm[i]));
        }
    }

    // ---- FASTCONV: bake one F32 copy of each F16 conv kernel and re-point the
    // c->tensors map entry to it. kokoro feeds these kernels straight to
    // ggml_conv_1d, which casts an F16 kernel → F32 inside EVERY graph when the
    // activations are F32; baking that cast once makes it a no-op, bitwise-equal.
    // The ConvTranspose1d upsamples (dec.gen.ups.{0,1}) use the SEPARATE F32
    // `ups_w_perm` buffers built just above (from the original F16 src), so swapping
    // their c->tensors entry is harmless (that entry is unused afterwards). The
    // depthwise-convt path (core_convt::convt1d_depthwise_2x_k3) casts F16→F32
    // internally too, so an already-F32 base just skips that cast. Gated
    // KOKORO_FASTCONV (default on — numerically equivalent).
    {
        const char* env = getenv("KOKORO_FASTCONV");
        const bool fc_on = !env || env[0] != '0';
        std::vector<ggml_tensor*> kernels;
        for (auto& kv : c->tensors) {
            ggml_tensor* w = kv.second;
            if (w && w->type == GGML_TYPE_F16 && ggml_n_dims(w) == 3)
                kernels.push_back(w);
        }
        c->fc.bake(c->backend, kernels, fc_on);
        int swapped = 0;
        for (auto& kv : c->tensors) {
            ggml_tensor* baked = c->fc.get(kv.second);
            if (baked != kv.second) {
                kv.second = baked;
                swapped++;
            }
        }
        if (getenv("KOKORO_FASTCONV_DEBUG"))
            kokoro_log(KOKORO_LOG_DEBUG, "kokoro: FASTCONV %s: %zu F16 3D conv kernels, %d baked+swapped",
                       fc_on ? "ON" : "OFF", kernels.size(), swapped);
    }

    // ---- Schedulers ----
    {
        ggml_backend_t backends[2];
        int n_be = 0;
        backends[n_be++] = c->backend;
        if (c->backend_cpu && c->backend_cpu != c->backend)
            backends[n_be++] = c->backend_cpu;
        // 256k nodes — predictor has 4 bidir LSTMs at T_frames up to ~1500
        // (~54k LSTM nodes) plus the decoder graph; the upper bound is
        // generous to accommodate AdainResBlk1d + AdaIN expansion.
        c->sched = ggml_backend_sched_new(backends, nullptr, n_be, /*graph_size=*/262144,
                                          /*parallel=*/false, /*op_offload=*/false);
        if (!c->sched) {
            kokoro_set_error("failed to allocate main scheduler");
            kokoro_free(c);
            return nullptr;
        }

    }
    c->compute_meta.resize(ggml_tensor_overhead() * 262144 + ggml_graph_overhead_custom(262144, false));

    if (params.verbosity >= 1) {
        kokoro_log(KOKORO_LOG_INFO, "kokoro: loaded %zu tensors from '%s'", c->tensors.size(), path_model);
    }
    return c;
}

extern "C" enum kokoro_status kokoro_load_voice_pack(struct kokoro_context* ctx, const char* path) {
    if (!ctx || !path) {
        kokoro_set_error("null context or path");
        return KOKORO_STATUS_INVALID_PARAMS;
    }

    // Read voice metadata: kokoro_voice.{name,max_phonemes,style_dim}.
    {
        gguf_context* g = core_gguf::open_metadata(path);
        if (!g) {
            kokoro_set_error("failed to read voice pack '%s'", path);
            return KOKORO_STATUS_LOAD_FAILED;
        }

        std::string arch = core_gguf::kv_str(g, "general.architecture", "");
        if (arch != "kokoro-voice") {
            kokoro_set_error("voice pack '%s' has architecture='%s' (want 'kokoro-voice')", path, arch.c_str());
            gguf_free(g);
            return KOKORO_STATUS_LOAD_FAILED;
        }

        kokoro_voice_pack vp;
        vp.name = core_gguf::kv_str(g, "kokoro_voice.name", "");
        vp.max_phonemes = core_gguf::kv_u32(g, "kokoro_voice.max_phonemes", 0);
        vp.style_dim = core_gguf::kv_u32(g, "kokoro_voice.style_dim", 0);
        gguf_free(g);

        // Replace any previously-loaded pack.
        if (ctx->vp.vp_buf_w)
            ggml_backend_buffer_free(ctx->vp.vp_buf_w);
        if (ctx->vp.vp_ctx_w)
            ggml_free(ctx->vp.vp_ctx_w);
        ctx->vp = std::move(vp);
        ctx->vp_loaded = false;
    }

    // Load the single F32 tensor `voice.pack`. Use the main `backend` so
    // ggml_get_rows / ggml_view_2d access works without a backend hop
    // during graph build.
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, ctx->backend, "kokoro.voice", wl)) {
        kokoro_set_error("failed to load voice pack tensors from '%s'", path);
        return KOKORO_STATUS_LOAD_FAILED;
    }

    auto it = wl.tensors.find("voice.pack");
    if (it == wl.tensors.end() || !it->second) {
        kokoro_set_error("voice pack '%s' missing 'voice.pack' tensor", path);
        ggml_backend_buffer_free(wl.buf);
        ggml_free(wl.ctx);
        return KOKORO_STATUS_LOAD_FAILED;
    }

    ggml_tensor* pack_ptr = it->second; // save before move invalidates iterator
    ctx->vp.tensors = std::move(wl.tensors);
    ctx->vp.pack = pack_ptr;
    ctx->vp.vp_ctx_w = wl.ctx;
    ctx->vp.vp_buf_w = wl.buf;
    ctx->vp_loaded = true;

    if (ctx->params.verbosity >= 1) {
        kokoro_log(KOKORO_LOG_INFO, "kokoro: voice '%s' (max_phonemes=%u style_dim=%u) loaded from '%s'",
                   ctx->vp.name.c_str(), ctx->vp.max_phonemes, ctx->vp.style_dim, path);
    }
    return KOKORO_STATUS_OK;
}

extern "C" enum kokoro_status kokoro_set_language(struct kokoro_context* ctx, const char* espeak_lang) {
    if (!ctx || !espeak_lang) {
        kokoro_set_error("null context or language");
        return KOKORO_STATUS_INVALID_PARAMS;
    }
    ctx->espeak_lang = espeak_lang;
    return KOKORO_STATUS_OK;
}

extern "C" void kokoro_phoneme_cache_clear(struct kokoro_context* ctx) {
    if (!ctx)
        return;
    ctx->phon_cache.clear();
}

extern "C" int32_t* kokoro_phonemes_to_ids(struct kokoro_context* ctx, const char* phonemes, int* out_n) {
    if (out_n)
        *out_n = 0;
    if (!ctx || !phonemes)
        return nullptr;

    // The vocab is 178 IPA symbols, each typically 1 codepoint (1-4 UTF-8
    // bytes). We greedy-tokenise: at each position, try the longest
    // matching token first (combining marks like ̃ or ː are 2 bytes and
    // attach to the previous letter, so they tokenise as their own ids
    // when they appear standalone). Unknown codepoints emit pad_id and a
    // callback warning.
    std::vector<int32_t> ids;
    ids.reserve(std::strlen(phonemes));

    auto utf8_len = [](unsigned char c) -> int {
        if ((c & 0x80) == 0)
            return 1;
        if ((c & 0xE0) == 0xC0)
            return 2;
        if ((c & 0xF0) == 0xE0)
            return 3;
        if ((c & 0xF8) == 0xF0)
            return 4;
        return 1; // invalid leading byte — consume one byte and move on
    };

    const char* p = phonemes;
    while (*p) {
        // Try lookahead: 4-byte → 3-byte → 2-byte → 1-byte greedy.
        // Combining marks following a base letter can form 2-codepoint
        // tokens (e.g. "̃" follows a base) — we don't try to combine them
        // because StyleTTS2's vocab stores them as standalone entries.
        int best_len = 0;
        int best_id = -1;
        for (int try_len = 4; try_len >= 1; try_len--) {
            // Fast path: skip if the current byte's UTF-8 length forbids it.
            const int real_len = utf8_len((unsigned char)*p);
            if (try_len > real_len)
                continue;
            std::string tok(p, (size_t)try_len);
            auto it = ctx->vocab.token_to_id.find(tok);
            if (it != ctx->vocab.token_to_id.end()) {
                best_len = try_len;
                best_id = it->second;
                break;
            }
        }
        if (best_id >= 0) {
            ids.push_back(best_id);
            p += best_len;
        } else {
            // Match the reference KModel.forward() behaviour: unknown
            // phonemes are dropped (Python uses
            // `filter(lambda i: i is not None, map(vocab.get, phonemes))`).
            // Emitting pad here would diverge from the reference and
            // poison every downstream stage's diff.
            const int n = utf8_len((unsigned char)*p);
            if (ctx->params.verbosity >= 1) {
                std::string bad(p, (size_t)n);
                kokoro_log(KOKORO_LOG_WARN, "kokoro: unknown phoneme '%s' skipped", bad.c_str());
            }
            p += n;
        }
    }

    int32_t* out = (int32_t*)std::malloc(ids.size() * sizeof(int32_t));
    if (!out) {
        kokoro_set_error("failed to allocate phoneme token buffer");
        return nullptr;
    }
    std::memcpy(out, ids.data(), ids.size() * sizeof(int32_t));
    if (out_n)
        *out_n = (int)ids.size();
    return out;
}

extern "C" float* kokoro_synthesize_phonemes(struct kokoro_context* ctx, const char* phonemes, int* out_n_samples) {
    if (out_n_samples)
        *out_n_samples = 0;
    if (!ctx || !phonemes || !*phonemes) {
        kokoro_set_error("invalid argument to kokoro_synthesize_phonemes");
        return nullptr;
    }
    if (!ctx->vp_loaded) {
        kokoro_set_error("no voice pack loaded");
        return nullptr;
    }

    int n_ids = 0;
    int32_t* ids = nullptr;
    {
        kokoro_bench_stage _b("phoneme_tokenize");
        ids = kokoro_phonemes_to_ids(ctx, phonemes, &n_ids);
    }
    if (!ids || n_ids == 0) {
        std::free(ids);
        kokoro_set_error("empty phoneme tokenisation for '%s'", phonemes);
        return nullptr;
    }
    int n_audio = 0;
    float* audio = kokoro_synthesize_pipeline(ctx, ids, n_ids, &n_audio);
    std::free(ids);
    if (!audio || n_audio <= 0)
        return nullptr;
    if (out_n_samples)
        *out_n_samples = n_audio;
    return audio;
}

namespace {

// Strip ASCII whitespace from both ends.
void rstrip_inplace(std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace((unsigned char)s[b]))
        b++;
    while (e > b && std::isspace((unsigned char)s[e - 1]))
        e--;
    s = s.substr(b, e - b);
}

// libespeak-ng's state is process-global, so all access goes through one
// mutex. Init is one-shot; voice switches are sticky.
std::mutex g_espeak_mu;
bool g_espeak_inited = false;
bool g_espeak_init_failed = false; // sticky — don't keep retrying
std::string g_espeak_voice;


// Returns true on success and fills `out`.
bool phonemize_espeak_lib(const std::string& lang, const std::string& text, std::string& out) {
    std::lock_guard<std::mutex> g(g_espeak_mu);
    if (g_espeak_init_failed)
        return false;
    if (!g_espeak_inited) {
        const char* path = std::getenv("KOKORO_ESPEAK_DATA_PATH");
        int sr = espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, path,
                                   espeakINITIALIZE_PHONEME_IPA | espeakINITIALIZE_DONT_EXIT);
        if (sr < 0) {
            kokoro_log(KOKORO_LOG_ERROR, "kokoro: espeak_Initialize failed (data path=%s)",
                       path ? path : "<default>");
            g_espeak_init_failed = true;
            return false;
        }
        g_espeak_inited = true;
    }
    if (g_espeak_voice != lang) {
        if (espeak_SetVoiceByName(lang.c_str()) != EE_OK) {
            kokoro_log(KOKORO_LOG_ERROR, "kokoro: espeak_SetVoiceByName('%s') failed", lang.c_str());
            return false;
        }
        g_espeak_voice = lang;
    }
    out.clear();
    const void* tp = text.c_str();
    while (tp) {
        const char* chunk = espeak_TextToPhonemes(&tp, espeakCHARS_UTF8, espeakPHONEMES_IPA);
        if (chunk && *chunk) {
            if (!out.empty())
                out += ' ';
            out += chunk;
        }
    }
    rstrip_inplace(out);
    return !out.empty();
}



// Post-process Mandarin/Chinese phonemes from espeak-ng.  espeak emits
// tone numbers (1-5) after each syllable (e.g. "ni2xˈɑu3") which
// Kokoro's 178-symbol IPA tokenizer can't represent.  Strip them so
// the IPA tokens map cleanly.  Also strip numeric stress markers that
// espeak sometimes emits for CJK (e.g. "ˈ1" sequences).  (PLAN #56)
static void strip_cmn_tone_numbers(std::string& s) {
    std::string r;
    r.reserve(s.size());
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] >= '0' && s[i] <= '9')
            continue; // drop tone number
        r += s[i];
    }
    s = std::move(r);
}

static bool is_cmn_lang(const std::string& lang) {
    return lang == "cmn" || lang == "zh" || lang == "zh-cn" || lang == "zh_cn" || lang == "cmn-latn-pinyin";
}


bool phonemize_cached(kokoro_context* ctx, const std::string& lang, const std::string& text, std::string& out) {
    std::string key = lang;
    key.push_back('\0');
    key += text;
    if (ctx->phon_cache.lookup(key, out))
        return true;


    // Always use espeak-ng (direct linkage).
    bool ok = false;
    if (phonemize_espeak_lib(lang, text, out)) {
        kokoro_phon::strip_espeak_lang_markers(out);
        ok = true;
    }

    if (ok && !out.empty()) {
        if (is_cmn_lang(lang))
            strip_cmn_tone_numbers(out);
        ctx->phon_cache.insert(key, out);
        return true;
    }
    return false;
}

} // namespace





extern "C" float* kokoro_synthesize(struct kokoro_context* ctx, const char* text, int* out_n_samples) {
    if (out_n_samples)
        *out_n_samples = 0;
    if (!ctx || !text || !*text) {
        kokoro_set_error("invalid argument to kokoro_synthesize");
        return nullptr;
    }
    if (!ctx->vp_loaded) {
        kokoro_set_error("no voice pack loaded");
        return nullptr;
    }

    // Process-wide, one-shot quality warnings.
    static std::atomic<bool> warned_cmn{false};
    static std::atomic<bool> warned_ja_kanji{false};
    if (is_cmn_lang(ctx->espeak_lang) && !warned_cmn.exchange(true)) {
        kokoro_log(KOKORO_LOG_WARN,
                   "kokoro: Mandarin tone information is lost because espeak-ng tone numbers "
                   "are stripped and the model vocabulary has no tone slots");
    }
    if (!warned_ja_kanji.load() && (ctx->espeak_lang == "ja" || ctx->espeak_lang == "jp")) {
        // Check for CJK Unified Ideographs (U+4E00–U+9FFF)
        const auto* p = (const unsigned char*)text;
        while (*p) {
            if (p[0] >= 0xE4 && p[0] <= 0xE9 && p[1] && p[2]) {
                uint32_t cp = ((p[0] & 0x0F) << 12) | ((p[1] & 0x3F) << 6) | (p[2] & 0x3F);
                if (cp >= 0x4E00 && cp <= 0x9FFF) {
                    if (!warned_ja_kanji.exchange(true)) {
                        kokoro_log(KOKORO_LOG_WARN,
                                   "kokoro: Japanese text contains kanji; espeak-ng's kanji "
                                   "dictionary is incomplete");
                    }
                    break;
                }
                p += 3;
            } else if (*p < 0x80) {
                p++;
            } else if (*p < 0xE0) {
                p += 2;
            } else if (*p < 0xF0) {
                p += 3;
            } else {
                p += 4;
            }
        }
    }

    std::string phonemes;
    {
        kokoro_bench_stage _b("phonemize");
        if (!phonemize_cached(ctx, ctx->espeak_lang, text, phonemes)) {
            kokoro_set_error("phonemizer produced no output for '%s'", text);
            return nullptr;
        }
    }
    if (ctx->params.verbosity >= 1)
        kokoro_log(KOKORO_LOG_INFO, "kokoro: phonemes: '%s'", phonemes.c_str());
    return kokoro_synthesize_phonemes(ctx, phonemes.c_str(), out_n_samples);
}



extern "C" void kokoro_pcm_free(float* pcm) {
    std::free(pcm);
}

extern "C" int kokoro_sample_rate(const struct kokoro_context* ctx) {
    return ctx ? (int)ctx->hp.sample_rate : 0;
}

extern "C" void kokoro_set_n_threads(struct kokoro_context* ctx, int n_threads) {
    if (!ctx || n_threads <= 0)
        return;
    ctx->n_threads = n_threads;
    if (ctx->backend_cpu)
        ggml_backend_cpu_set_n_threads(ctx->backend_cpu, n_threads);
}

extern "C" enum kokoro_backend kokoro_get_backend(const struct kokoro_context* ctx) {
    return ctx ? ctx->selected_backend : KOKORO_BACKEND_AUTO;
}

extern "C" const char* kokoro_backend_name(const struct kokoro_context* ctx) {
    if (!ctx)
        return nullptr;
    switch (ctx->selected_backend) {
    case KOKORO_BACKEND_CPU:    return "CPU";
    case KOKORO_BACKEND_VULKAN: return "Vulkan";
    case KOKORO_BACKEND_METAL:  return "Metal";
    case KOKORO_BACKEND_CUDA:   return "CUDA";
    case KOKORO_BACKEND_AUTO:   break;
    }
    return "unknown";
}

// The duration-predictor output is scaled before rounding and clamping.
extern "C" void kokoro_set_length_scale(struct kokoro_context* ctx, float scale) {
    if (!ctx)
        return;
    if (scale < 0.25f)
        scale = 0.25f;
    if (scale > 4.0f)
        scale = 4.0f;
    ctx->params.length_scale = scale;
}

extern "C" void kokoro_free(struct kokoro_context* ctx) {
    if (!ctx)
        return;
    ctx->fc.free(); // FASTCONV baked kernels (before the backend is freed)
    if (ctx->gen_sched)
        ggml_backend_sched_free(ctx->gen_sched);
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->vp.vp_buf_w)
        ggml_backend_buffer_free(ctx->vp.vp_buf_w);
    if (ctx->vp.vp_ctx_w)
        ggml_free(ctx->vp.vp_ctx_w);
    if (ctx->buf_perm)
        ggml_backend_buffer_free(ctx->buf_perm);
    if (ctx->ctx_perm)
        ggml_free(ctx->ctx_perm);
    if (ctx->buf_w)
        ggml_backend_buffer_free(ctx->buf_w);
    if (ctx->ctx_w)
        ggml_free(ctx->ctx_w);
    if (ctx->backend && ctx->backend != ctx->backend_cpu)
        ggml_backend_free(ctx->backend);
    if (ctx->backend_cpu)
        ggml_backend_free(ctx->backend_cpu);
    delete ctx;
}

// ---------------------------------------------------------------------------
// New API: version, logging, error, audio struct, voice query
// ---------------------------------------------------------------------------

extern "C" const char * kokoro_version(void) {
    return g_version_str;
}

extern "C" const char * kokoro_last_error(void) {
    return g_last_error.c_str();
}

extern "C" void kokoro_log_set(kokoro_log_cb cb, void * user_data) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    g_log_cb = cb;
    g_log_ud = user_data;
}

extern "C" const char * kokoro_voice_name(const struct kokoro_context * ctx) {
    if (!ctx || !ctx->vp_loaded) return NULL;
    return ctx->vp.name.c_str();
}

extern "C" int kokoro_voice_max_phonemes(const struct kokoro_context * ctx) {
    if (!ctx || !ctx->vp_loaded) return 0;
    return (int)ctx->vp.max_phonemes;
}

extern "C" void kokoro_audio_free(struct kokoro_audio * audio) {
    if (!audio) return;
    std::free(audio->samples);
    audio->samples = NULL;
    audio->n_samples = 0;
    audio->sample_rate = 0;
    audio->channels = 0;
}

extern "C" enum kokoro_status kokoro_synthesize_audio(struct kokoro_context * ctx,
                                                      const char * text,
                                                      struct kokoro_audio * out) {
    if (!ctx || !text || !*text || !out) {
        kokoro_set_error("invalid argument to kokoro_synthesize_audio");
        return KOKORO_STATUS_INVALID_PARAMS;
    }
    if (!ctx->vp_loaded) {
        kokoro_set_error("no voice pack loaded");
        return KOKORO_STATUS_NO_VOICE;
    }
    // Free any previous contents
    kokoro_audio_free(out);
    g_last_error.clear();

    int n = 0;
    float * pcm = kokoro_synthesize(ctx, text, &n);
    if (!pcm || n <= 0) {
        if (g_last_error.empty())
            kokoro_set_error("text synthesis failed");
        return KOKORO_STATUS_SYNTH_FAILED;
    }
    out->samples = pcm;
    out->n_samples = n;
    out->sample_rate = (int)ctx->hp.sample_rate;
    out->channels = 1;
    return KOKORO_STATUS_OK;
}

extern "C" enum kokoro_status kokoro_synthesize_phonemes_audio(struct kokoro_context * ctx,
                                                               const char * phonemes,
                                                               struct kokoro_audio * out) {
    if (!ctx || !phonemes || !*phonemes || !out) {
        kokoro_set_error("invalid argument to kokoro_synthesize_phonemes_audio");
        return KOKORO_STATUS_INVALID_PARAMS;
    }
    if (!ctx->vp_loaded) {
        kokoro_set_error("no voice pack loaded");
        return KOKORO_STATUS_NO_VOICE;
    }
    kokoro_audio_free(out);
    g_last_error.clear();

    int n = 0;
    float * pcm = kokoro_synthesize_phonemes(ctx, phonemes, &n);
    if (!pcm || n <= 0) {
        if (g_last_error.empty())
            kokoro_set_error("phoneme synthesis failed");
        return KOKORO_STATUS_SYNTH_FAILED;
    }
    out->samples = pcm;
    out->n_samples = n;
    out->sample_rate = (int)ctx->hp.sample_rate;
    out->channels = 1;
    return KOKORO_STATUS_OK;
}
