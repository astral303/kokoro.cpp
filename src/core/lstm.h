// src/core/lstm.h — bidirectional LSTM helpers for ggml graphs.
//
// PyTorch nn.LSTM forward (single layer, single direction):
//
//   pre_t = W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh   ∈ R^{4H}
//   i, f, g, o = pre_t.split(H)                          (PyTorch gate order)
//   c_t = sigmoid(f) * c_{t-1} + sigmoid(i) * tanh(g)
//   h_t = sigmoid(o) * tanh(c_t)
//
// Storage convention used by every Kokoro / StyleTTS2 LSTM in the GGUF:
//
//   weight_ih_l0          ne = (input_size,  4 * hidden_size)   F16
//   weight_hh_l0          ne = (hidden_size, 4 * hidden_size)   F16
//   bias_ih_l0            ne = (4 * hidden_size,)               F32
//   bias_hh_l0            ne = (4 * hidden_size,)               F32
//   weight_ih_l0_reverse  same shape as weight_ih_l0 (bidir layer 0)
//   ... etc for hh + biases
//
// Both biases are added (PyTorch keeps them separate even though their
// sum is mathematically equivalent — we follow the same pattern so the
// graph is bit-equivalent to the reference dump).
//
// We optimise the per-step graph by hoisting the input projection out of
// the loop:  proj_x = (W_ih @ X) + b_ih, computed once for all T at
// once.  Per timestep we still need W_hh @ h_{t-1} (the recurrence is
// fundamentally sequential).  At t=0 the helper skips the W_hh matmul
// entirely (h_{-1} = 0), which matches PyTorch's default zero initial
// state and avoids needing an externally-zeroed buffer.
//
// Output assembly is functional: timestep columns are combined with a
// balanced concat tree. Avoid side-effecting cpy-to-view writes here:
// backend graph optimizers are free to transform the consumer of the
// destination tensor without preserving those implicit write dependencies.
//
// Bidirectional layers can optionally run both directions as GGML batch
// dimension ne[2]=2: stack forward X with a time-reversed gather of X,
// concat forward/reverse weights and biases on ne[2], then share one
// recurrent step's matmuls and gate elementwise ops across both
// directions. Final (2H, T) is assembled functionally (forward at t,
// reverse at T-1-t). This is intended for GPU backends where dispatch
// overhead dominates; CPU keeps the historical dual-unidir path.

#pragma once

#include "matmul.h"
#include "ggml.h"

#include <cstddef>
#include <vector>

namespace core_lstm {

// Build I32 indices [T-1, T-2, ..., 0] with ordinary GGML ops (arange →
// negate/offset → cast). Used for time-reversing X and for placing the
// reverse-direction hidden states back into original time order.
static inline ggml_tensor* reverse_time_indices(ggml_context* ctx, int T) {
    // F32 [0, 1, ..., T-1]; integers in this range are exact in F32.
    ggml_tensor* fwd = ggml_arange(ctx, 0.0f, (float)T, 1.0f);
    // Scalar (T-1) via arange — arange(0,1,1) is 0, not 1.
    ggml_tensor* t_last = ggml_arange(ctx, (float)(T - 1), (float)T, 1.0f);
    ggml_tensor* rev_f = ggml_add(ctx, ggml_scale(ctx, fwd, -1.0f), t_last);
    return ggml_cast(ctx, rev_f, GGML_TYPE_I32);
}

// Concatenate a pair of 2D tensors along ne[2] after promoting each to
// shape (ne0, ne1, 1). Used for forward/reverse weight and bias packing.
static inline ggml_tensor* concat_dir2(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b) {
    ggml_tensor* a3 = ggml_reshape_3d(ctx, a, a->ne[0], a->ne[1], 1);
    ggml_tensor* b3 = ggml_reshape_3d(ctx, b, b->ne[0], b->ne[1], 1);
    return ggml_concat(ctx, a3, b3, /*dim=*/2);
}

// Balanced concat along `dim` — logarithmic depth, no cpy-to-view writes.
static inline ggml_tensor* concat_tree(ggml_context* ctx, std::vector<ggml_tensor*>& columns, int dim) {
    while (columns.size() > 1) {
        std::vector<ggml_tensor*> next;
        next.reserve((columns.size() + 1) / 2);
        for (size_t i = 0; i < columns.size(); i += 2) {
            if (i + 1 < columns.size())
                next.push_back(ggml_concat(ctx, columns[i], columns[i + 1], dim));
            else
                next.push_back(columns[i]);
        }
        columns = std::move(next);
    }
    return columns[0];
}

// Single-direction LSTM forward over T timesteps.
//
//   ctx          per-graph ggml context (no_alloc=true)
//   X            input, ne = (input_size, T)  F32
//   W_ih         ne = (input_size,  4H)        F16/F32
//   W_hh         ne = (hidden_size, 4H)        F16/F32
//   b_ih, b_hh   ne = (4H,)                     F32
//   H            hidden_size
//   reverse      iterate t = T-1 .. 0 (use the *_reverse weights)
//
// Returns the LSTM output as a contiguous (H, T) F32 tensor.
static inline ggml_tensor* lstm_unidir(ggml_context* ctx, ggml_tensor* X, ggml_tensor* W_ih, ggml_tensor* W_hh,
                                       ggml_tensor* b_ih, ggml_tensor* b_hh, int H, bool reverse) {
    const int T = (int)X->ne[1];
    const int H4 = 4 * H;

    // Input projection over all T timesteps at once.
    // proj_x ne = (4H, T)
    ggml_tensor* proj_x = core_ggml::mul_mat(ctx, W_ih, X);
    proj_x = ggml_add(ctx, proj_x, b_ih);

    ggml_tensor* h = nullptr;
    ggml_tensor* c = nullptr;
    std::vector<ggml_tensor*> columns((size_t)T);

    const size_t row_stride_4h = proj_x->nb[1];
    const size_t f32_size = ggml_type_size(GGML_TYPE_F32);

    const int t0 = reverse ? T - 1 : 0;
    const int dt = reverse ? -1 : 1;
    for (int step = 0; step < T; step++) {
        const int t = t0 + step * dt;

        // proj_x slice for this timestep: (4H, 1) F32, contiguous.
        ggml_tensor* px = ggml_view_2d(ctx, proj_x, H4, 1, row_stride_4h, (size_t)t * row_stride_4h);

        // pre = px + (W_hh @ h + b_hh) — at step 0 the W_hh@0 term vanishes.
        ggml_tensor* pre;
        if (h) {
            ggml_tensor* ph = core_ggml::mul_mat(ctx, W_hh, h); // (4H, 1)
            ph = ggml_add(ctx, ph, b_hh);
            pre = ggml_add(ctx, px, ph);
        } else {
            pre = ggml_add(ctx, px, b_hh);
        }

        // Split (4H, 1) into 4 gates of shape (H, 1).  Gate order is
        // PyTorch's (i, f, g, o), each H-wide stripe along ne[0].
        const size_t pre_stride = pre->nb[1];
        ggml_tensor* gi = ggml_view_2d(ctx, pre, H, 1, pre_stride, (size_t)0 * H * f32_size);
        ggml_tensor* gf_gate = ggml_view_2d(ctx, pre, H, 1, pre_stride, (size_t)1 * H * f32_size);
        ggml_tensor* gg = ggml_view_2d(ctx, pre, H, 1, pre_stride, (size_t)2 * H * f32_size);
        ggml_tensor* go = ggml_view_2d(ctx, pre, H, 1, pre_stride, (size_t)3 * H * f32_size);

        ggml_tensor* sig_i = ggml_sigmoid(ctx, gi);
        ggml_tensor* sig_f = ggml_sigmoid(ctx, gf_gate);
        ggml_tensor* sig_o = ggml_sigmoid(ctx, go);
        ggml_tensor* tanh_g = ggml_tanh(ctx, gg);

        // c_t = sigmoid(f) * c_{t-1} + sigmoid(i) * tanh(g)
        if (c) {
            c = ggml_add(ctx, ggml_mul(ctx, sig_f, c), ggml_mul(ctx, sig_i, tanh_g));
        } else {
            c = ggml_mul(ctx, sig_i, tanh_g);
        }
        // h_t = sigmoid(o) * tanh(c_t)
        h = ggml_mul(ctx, sig_o, ggml_tanh(ctx, c));

        columns[(size_t)t] = h;
    }

    return concat_tree(ctx, columns, /*dim=*/1);
}

// Bidirectional LSTM with forward and reverse directions packed on ne[2].
//
//   X_batch      = concat(X, gather(X, rev_idx), dim=2)     (I, T, 2)
//   W_ih_batch   = concat(W_ih_f, W_ih_r, dim=2)            (I, 4H, 2)
//   ... same for W_hh / biases
//   proj_x       = W_ih_batch @ X_batch + b_ih_batch        (4H, T, 2)
//   recurrent h,c shapes (H, 1, 2); one matmul / gate op serves both dirs
//   output       = concat(fwd[t], rev[T-1-t], dim=0)        (2H, T)
//
// Weight packing is graph-concatenated (no persistent prepack). Reverse
// indices are built with GGML ops so callers do not upload I32 buffers.
static inline ggml_tensor* lstm_bidir_batched(ggml_context* ctx, ggml_tensor* X, ggml_tensor* W_ih_f, ggml_tensor* W_hh_f,
                                              ggml_tensor* b_ih_f, ggml_tensor* b_hh_f, ggml_tensor* W_ih_r,
                                              ggml_tensor* W_hh_r, ggml_tensor* b_ih_r, ggml_tensor* b_hh_r, int H) {
    const int T = (int)X->ne[1];
    const int I = (int)X->ne[0];
    const int H4 = 4 * H;
    const size_t f32_size = ggml_type_size(GGML_TYPE_F32);

    ggml_tensor* rev_idx = reverse_time_indices(ctx, T);
    ggml_tensor* X_rev = ggml_get_rows(ctx, X, rev_idx); // (I, T)

    ggml_tensor* X_b = ggml_concat(ctx, ggml_reshape_3d(ctx, X, I, T, 1), ggml_reshape_3d(ctx, X_rev, I, T, 1),
                                   /*dim=*/2); // (I, T, 2)

    ggml_tensor* W_ih = concat_dir2(ctx, W_ih_f, W_ih_r); // (I, 4H, 2)
    ggml_tensor* W_hh = concat_dir2(ctx, W_hh_f, W_hh_r); // (H, 4H, 2)
    ggml_tensor* b_ih = concat_dir2(ctx, b_ih_f, b_ih_r); // (4H, 1, 2)
    ggml_tensor* b_hh = concat_dir2(ctx, b_hh_f, b_hh_r); // (4H, 1, 2)

    // Batched input projection: (4H, T, 2)
    ggml_tensor* proj_x = core_ggml::mul_mat(ctx, W_ih, X_b);
    proj_x = ggml_add(ctx, proj_x, b_ih);

    ggml_tensor* h = nullptr;
    ggml_tensor* c = nullptr;
    std::vector<ggml_tensor*> columns((size_t)T);

    for (int step = 0; step < T; step++) {
        // Both directions share step: reverse X is already time-reversed.
        ggml_tensor* px = ggml_view_3d(ctx, proj_x, H4, 1, 2, proj_x->nb[1], proj_x->nb[2],
                                       (size_t)step * proj_x->nb[1]); // (4H, 1, 2)

        ggml_tensor* pre;
        if (h) {
            ggml_tensor* ph = core_ggml::mul_mat(ctx, W_hh, h); // (4H, 1, 2)
            ph = ggml_add(ctx, ph, b_hh);
            pre = ggml_add(ctx, px, ph);
        } else {
            pre = ggml_add(ctx, px, b_hh);
        }

        const size_t pre_nb1 = pre->nb[1];
        const size_t pre_nb2 = pre->nb[2];
        ggml_tensor* gi = ggml_view_3d(ctx, pre, H, 1, 2, pre_nb1, pre_nb2, (size_t)0 * H * f32_size);
        ggml_tensor* gf_gate = ggml_view_3d(ctx, pre, H, 1, 2, pre_nb1, pre_nb2, (size_t)1 * H * f32_size);
        ggml_tensor* gg = ggml_view_3d(ctx, pre, H, 1, 2, pre_nb1, pre_nb2, (size_t)2 * H * f32_size);
        ggml_tensor* go = ggml_view_3d(ctx, pre, H, 1, 2, pre_nb1, pre_nb2, (size_t)3 * H * f32_size);

        ggml_tensor* sig_i = ggml_sigmoid(ctx, gi);
        ggml_tensor* sig_f = ggml_sigmoid(ctx, gf_gate);
        ggml_tensor* sig_o = ggml_sigmoid(ctx, go);
        ggml_tensor* tanh_g = ggml_tanh(ctx, gg);

        if (c) {
            c = ggml_add(ctx, ggml_mul(ctx, sig_f, c), ggml_mul(ctx, sig_i, tanh_g));
        } else {
            c = ggml_mul(ctx, sig_i, tanh_g);
        }
        h = ggml_mul(ctx, sig_o, ggml_tanh(ctx, c)); // (H, 1, 2)

        columns[(size_t)step] = h;
    }

    ggml_tensor* ht = concat_tree(ctx, columns, /*dim=*/1); // (H, T, 2)
    if (!ggml_is_contiguous(ht))
        ht = ggml_cont(ctx, ht);

    // Batch 0 is forward in natural time; batch 1 is reverse in step order
    // (step s ↔ original time T-1-s), so gather with the same rev_idx.
    ggml_tensor* fwd = ggml_view_2d(ctx, ht, H, T, ht->nb[1], 0);
    ggml_tensor* bwd_steps = ggml_view_2d(ctx, ht, H, T, ht->nb[1], ht->nb[2]);
    if (!ggml_is_contiguous(fwd))
        fwd = ggml_cont(ctx, fwd);
    if (!ggml_is_contiguous(bwd_steps))
        bwd_steps = ggml_cont(ctx, bwd_steps);
    ggml_tensor* bwd = ggml_get_rows(ctx, bwd_steps, rev_idx); // (H, T)

    return ggml_concat(ctx, fwd, bwd, /*dim=*/0); // (2H, T)
}

// Bidirectional LSTM. Default path runs two unidirectional passes then
// concatenates along the feature dim: output ne = (2H, T) with forward H
// first, then backward H. When batch_dirs is true, both directions share
// GGML batch dimension 2 (see lstm_bidir_batched).
static inline ggml_tensor* lstm_bidir(ggml_context* ctx, ggml_tensor* X, ggml_tensor* W_ih_f, ggml_tensor* W_hh_f,
                                      ggml_tensor* b_ih_f, ggml_tensor* b_hh_f, ggml_tensor* W_ih_r,
                                      ggml_tensor* W_hh_r, ggml_tensor* b_ih_r, ggml_tensor* b_hh_r, int H,
                                      bool batch_dirs = false) {
    if (batch_dirs)
        return lstm_bidir_batched(ctx, X, W_ih_f, W_hh_f, b_ih_f, b_hh_f, W_ih_r, W_hh_r, b_ih_r, b_hh_r, H);

    ggml_tensor* fwd = lstm_unidir(ctx, X, W_ih_f, W_hh_f, b_ih_f, b_hh_f, H, /*reverse=*/false);
    ggml_tensor* bwd = lstm_unidir(ctx, X, W_ih_r, W_hh_r, b_ih_r, b_hh_r, H, /*reverse=*/true);
    return ggml_concat(ctx, fwd, bwd, /*dim=*/0);
}

} // namespace core_lstm
