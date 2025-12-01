/**
 * Variable-Length Flash Attention Forward Kernel
 *
 * Implements Flash Attention for packed (varlen) sequences where queries,
 * keys, and values are stored without padding. Each thread handles one
 * (batch, head, query_position) tuple using online softmax computation.
 *
 * Input layout:
 *   Q: [total_q, nheads, headdim]
 *   K: [total_k, nheads_k, headdim]
 *   V: [total_k, nheads_k, headdim]
 *
 * Output layout:
 *   O: [total_q, nheads, headdim]
 *   L: [nheads, total_q]
 *
 * Features:
 * - Causal masking: only attend to positions <= current position
 * - Sliding window: only attend to positions within window
 * - GQA/MQA: nheads_k can be different from nheads
 * - Softcap: tanh-based logit capping
 * - ALiBi: attention with linear biases for positional encoding
 *
 * Grid configuration (from Python wrapper):
 *   grid = (total_q * nheads, 1, 1)
 *   threadgroup = (threads_per_group, 1, 1)
 */

uint global_elem = thread_position_in_grid.x;

int total_threads = total_q * nheads;
if (global_elem >= uint(total_threads)) {
    return;
}

int head_idx = global_elem % nheads;
int global_q_pos = global_elem / nheads;

if (global_q_pos >= total_q || batch_size <= 0) {
    return;
}

// Binary search batch index for this query position
int lo = 0;
int hi = batch_size;
while (lo < hi) {
    int mid = (lo + hi) / 2;
    int mid_end = cu_seqlens_q[mid + 1];
    if (global_q_pos >= mid_end) {
        lo = mid + 1;
    } else {
        hi = mid;
    }
}
int batch_idx = min(lo, batch_size - 1);

int q_seq_start = cu_seqlens_q[batch_idx];
int q_seq_end = cu_seqlens_q[batch_idx + 1];
int seq_len_q = q_seq_end - q_seq_start;

if (seq_len_q <= 0) {
    int o_offset = (global_q_pos * nheads + head_idx) * headdim;
    int l_offset = head_idx * total_q + global_q_pos;
    for (int d = 0; d < headdim; ++d) {
        O[o_offset + d] = T(0);
    }
    L[l_offset] = -INFINITY;
    return;
}

int local_q_pos = global_q_pos - q_seq_start;
if (local_q_pos < 0 || local_q_pos >= seq_len_q) {
    int o_offset = (global_q_pos * nheads + head_idx) * headdim;
    int l_offset = head_idx * total_q + global_q_pos;
    for (int d = 0; d < headdim; ++d) {
        O[o_offset + d] = T(0);
    }
    L[l_offset] = -INFINITY;
    return;
}

int kv_ratio = max(1, nheads / max(1, nheads_k));
int kv_head = head_idx / kv_ratio;
kv_head = clamp(kv_head, 0, max(0, nheads_k - 1));
int kv_head_offset = kv_head * headdim;
int kv_row_stride = max(1, nheads_k) * headdim;
int safe_page_size = (page_size > 0) ? page_size : 1;
int kv_page_stride = safe_page_size * kv_row_stride;
int page_table_row_offset = batch_idx * max_pages_per_seq;
bool paged_kv_enabled = (use_paged_kv != 0);
int log2_page = log2_page_size;
int page_mask = (log2_page >= 0) ? ((1 << log2_page) - 1) : 0;

int q_offset = (global_q_pos * nheads + head_idx) * headdim;
int o_offset = q_offset;
int l_offset = head_idx * total_q + global_q_pos;

T q_local[256];
for (int d = 0; d < headdim; ++d) {
    q_local[d] = Q[q_offset + d];
}

float max_score = -INFINITY;
float sum_exp = 0.0f;
float out_local[256];
for (int d = 0; d < headdim; ++d) {
    out_local[d] = 0.0f;
}

int k_seq_start = cu_seqlens_k[batch_idx];
int k_seq_end = cu_seqlens_k[batch_idx + 1];
int seq_len_k = k_seq_end - k_seq_start;

if (seq_len_k <= 0) {
    for (int d = 0; d < headdim; ++d) {
        O[o_offset + d] = T(0);
    }
    L[l_offset] = -INFINITY;
    return;
}

int k_begin = 0;
int k_end = seq_len_k;

if (causal != 0) {
    k_end = min(k_end, local_q_pos + 1);
}
if (window_left >= 0) {
    k_begin = max(k_begin, local_q_pos - window_left);
}
if (window_right >= 0) {
    k_end = min(k_end, local_q_pos + window_right + 1);
}

if (k_begin < 0) {
    k_begin = 0;
}
if (k_end > seq_len_k) {
    k_end = seq_len_k;
}

if (k_begin >= k_end) {
    for (int d = 0; d < headdim; ++d) {
        O[o_offset + d] = T(0);
    }
    L[l_offset] = -INFINITY;
    return;
}

for (int local_k = k_begin; local_k < k_end; ++local_k) {
    int global_k_pos = k_seq_start + local_k;
    int k_offset;
    int v_offset;
    if (paged_kv_enabled) {
        if (max_pages_per_seq <= 0) {
            continue;
        }

        int page_idx;
        int page_offset;
        if (log2_page >= 0) {
            page_idx = local_k >> log2_page;
            page_offset = local_k & page_mask;
        } else {
            page_idx = local_k / safe_page_size;
            page_offset = local_k % safe_page_size;
        }

        if (page_idx >= max_pages_per_seq) {
            continue;
        }

        int table_entry = page_table[page_table_row_offset + page_idx];
        if (table_entry < 0) {
            continue;
        }

        int page_base = table_entry * kv_page_stride;
        int row_offset = page_offset * kv_row_stride;
        k_offset = page_base + row_offset + kv_head_offset;
        v_offset = k_offset;
    } else {
        k_offset = global_k_pos * kv_row_stride + kv_head_offset;
        v_offset = k_offset;
    }

    float score = 0.0f;
    for (int d = 0; d < headdim; ++d) {
        score += float(q_local[d]) * float(K[k_offset + d]);
    }
    score *= softmax_scale;

    if (softcap > 0.0f) {
        score = softcap * tanh(score / softcap);
    }

    // Apply ALiBi bias if enabled (use local positions relative to segment)
    if (use_alibi != 0) {
        int slope_idx = (alibi_batch_stride > 0) ? batch_idx * alibi_batch_stride + head_idx : head_idx;
        float slope = alibi_slopes[slope_idx];
        // ALiBi bias: slope * distance, where distance is computed based on causal mode
        int distance;
        if (causal != 0) {
            distance = local_k - local_q_pos;  // Causal: negative for positions before query
        } else {
            distance = -abs(local_q_pos - local_k);  // Non-causal: always negative or zero
        }
        score += slope * float(distance);
    }

    float new_max = max(max_score, score);
    float exp_diff = exp(max_score - new_max);
    float exp_score = exp(score - new_max);

    sum_exp = sum_exp * exp_diff + exp_score;
    for (int d = 0; d < headdim; ++d) {
        out_local[d] = out_local[d] * exp_diff;
    }

    float exp_score_for_v = exp_score;
    if (dropout_p > 0.0f) {
        float rand_val = philox_uniform(
            local_q_pos,
            local_k,
            batch_idx,
            head_idx,
            uint(philox_seed),
            uint(philox_offset)
        );
        if (rand_val < dropout_p) {
            exp_score_for_v = 0.0f;
        } else {
            exp_score_for_v = exp_score / (1.0f - dropout_p);
        }
    }

    for (int d = 0; d < headdim; ++d) {
        out_local[d] += exp_score_for_v * float(V[v_offset + d]);
    }

    max_score = new_max;
}

if (sum_exp <= 0.0f) {
    for (int d = 0; d < headdim; ++d) {
        O[o_offset + d] = T(0);
    }
    L[l_offset] = -INFINITY;
    return;
}

float norm = 1.0f / sum_exp;
for (int d = 0; d < headdim; ++d) {
    O[o_offset + d] = T(out_local[d] * norm);
}

L[l_offset] = max_score + log(sum_exp);
