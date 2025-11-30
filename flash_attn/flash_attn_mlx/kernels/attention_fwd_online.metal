/**
 * Flash Attention Forward Kernel (Online Softmax)
 *
 * This kernel implements the forward pass of Flash Attention using
 * online softmax computation for numerical stability and memory efficiency.
 *
 * Uses per-query parallelization - each thread handles one (batch, head, query_pos) tuple.
 * Optimized for clarity and correctness; a tiled version for better memory access
 * patterns will be added in a later optimization phase.
 *
 * Input layout: [batch, seqlen, nheads, headdim]
 * Output layout: [batch, seqlen, nheads, headdim] for O, [batch, nheads, seqlen] for L
 *
 * Features:
 * - Causal masking: only attend to positions <= current position
 * - Sliding window: only attend to positions within (query_pos - window_left, query_pos + window_right)
 * - GQA/MQA: nheads_k can be different from nheads (nheads must be divisible by nheads_k)
 * - Softcap: tanh-based logit capping for numerical stability
 * - ALiBi: attention with linear biases for positional encoding
 * - Dropout: attention weight dropout with Philox RNG
 *
 * Algorithm based on:
 * - FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning
 * - Online normalizer calculation for softmax (Milakov & Gimelshein, 2018)
 */

// Thread position
uint elem = thread_position_in_grid.x;

// Compute indices
int batch = elem / (seqlen_q * nheads);
int remainder = elem % (seqlen_q * nheads);
int query_pos = remainder / nheads;
int head = remainder % nheads;

if (batch >= batch_size || query_pos >= seqlen_q) {
    return;
}

// KV head for GQA/MQA
int kv_head = head / (nheads / nheads_k);

// Base offsets for [batch, seqlen, nheads, headdim] layout
int q_offset = batch * seqlen_q * nheads * headdim + query_pos * nheads * headdim + head * headdim;
int kv_batch_offset = batch * seqlen_k * nheads_k * headdim;
int o_offset = q_offset;
int lse_offset = batch * nheads * seqlen_q + head * seqlen_q + query_pos;
int kv_head_offset = kv_head * headdim;
int kv_row_stride = nheads_k * headdim;
int safe_page_size = (page_size > 0) ? page_size : 1;
int kv_page_stride = safe_page_size * kv_row_stride;
int page_table_row_offset = batch * max_pages_per_seq;
bool paged_kv_enabled = (use_paged_kv != 0);
int log2_page = log2_page_size;
int page_mask = (log2_page >= 0) ? ((1 << log2_page) - 1) : 0;

// Load query into registers
T q_local[256];  // Max headdim
for (int d = 0; d < headdim; d++) {
    q_local[d] = Q[q_offset + d];
}

// Online softmax state
float max_score = -INFINITY;
float sum_exp = 0.0f;
float out_local[256];  // Max headdim
for (int d = 0; d < headdim; d++) {
    out_local[d] = 0.0f;
}

// Compute key range for iteration
// Start: max(0, query_pos - window_left) if window_left >= 0, else 0
int k_start = (window_left >= 0) ? max(0, query_pos - window_left) : 0;

// End: min(seqlen_k, query_pos + window_right + 1) if window_right >= 0
//      For causal, also limit to query_pos + 1
int k_end = seqlen_k;
if (causal) {
    k_end = min(k_end, query_pos + 1);
}
if (window_right >= 0) {
    k_end = min(k_end, query_pos + window_right + 1);
}

for (int key_pos = k_start; key_pos < k_end; key_pos++) {
    int k_offset;
    int v_offset;
    if (paged_kv_enabled) {
        if (max_pages_per_seq <= 0) {
            continue;
        }

        int page_idx;
        int page_offset;
        if (log2_page >= 0) {
            page_idx = key_pos >> log2_page;
            page_offset = key_pos & page_mask;
        } else {
            page_idx = key_pos / safe_page_size;
            page_offset = key_pos % safe_page_size;
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
        k_offset = kv_batch_offset + key_pos * kv_row_stride + kv_head_offset;
        v_offset = k_offset;
    }

    float score = 0.0f;
    for (int d = 0; d < headdim; d++) {
        score += float(q_local[d]) * float(K[k_offset + d]);
    }
    score *= scale;

    // Apply softcap if enabled
    if (softcap > 0.0f) {
        score = softcap * tanh(score / softcap);
    }

    // Apply ALiBi bias if enabled
    if (use_alibi != 0) {
        int slope_idx = (alibi_batch_stride > 0) ? batch * alibi_batch_stride + head : head;
        float slope = alibi_slopes[slope_idx];
        // ALiBi bias: slope * distance, where distance is computed based on causal mode
        int distance;
        if (causal != 0) {
            distance = key_pos - query_pos;  // Causal: negative for positions before query
        } else {
            distance = -abs(query_pos - key_pos);  // Non-causal: always negative or zero
        }
        score += slope * float(distance);
    }

    // Online softmax update
    float new_max = max(max_score, score);
    float exp_diff = exp(max_score - new_max);
    float exp_score = exp(score - new_max);

    // Rescale running sum and add new score (BEFORE dropout for correct LSE)
    sum_exp = sum_exp * exp_diff + exp_score;

    // Rescale output accumulator
    for (int d = 0; d < headdim; d++) {
        out_local[d] = out_local[d] * exp_diff;
    }

    // Apply dropout to the value contribution (but not to sum_exp)
    float exp_score_for_v = exp_score;
    if (dropout_p > 0.0f) {
        // Philox 4x32-7 RNG (7 rounds) - inline implementation
        // Counter: (query_pos, key_pos, batch, head)
        // Key: (philox_seed_low, philox_seed_high)
        uint4 ctr = uint4(uint(query_pos), uint(key_pos), uint(batch), uint(head));
        uint2 key = uint2(uint(philox_seed), uint(philox_offset));

        // Philox constants (inline literals to avoid constant qualifier issues)
        // kPhiloxSA = 0xD2511F53, kPhiloxSB = 0xCD9E8D57
        // kPhilox10A = 0x9E3779B9, kPhilox10B = 0xBB67AE85

        // 7 rounds of Philox mixing
        for (int round = 0; round < 7; round++) {
            ulong prod0 = ulong(0xD2511F53) * ulong(ctr.x);
            ulong prod1 = ulong(0xCD9E8D57) * ulong(ctr.z);
            uint lo0 = uint(prod0);
            uint hi0 = uint(prod0 >> 32);
            uint lo1 = uint(prod1);
            uint hi1 = uint(prod1 >> 32);

            uint4 new_ctr;
            new_ctr.x = hi1 ^ ctr.y ^ key.x;
            new_ctr.y = lo1;
            new_ctr.z = hi0 ^ ctr.w ^ key.y;
            new_ctr.w = lo0;
            ctr = new_ctr;

            key.x += 0x9E3779B9;
            key.y += 0xBB67AE85;
        }

        // Convert first random uint32 to float in [0, 1)
        float rand_val = float(ctr.x) * 2.3283064365386963e-10f;

        if (rand_val < dropout_p) {
            exp_score_for_v = 0.0f;  // Drop this position
        } else {
            exp_score_for_v = exp_score / (1.0f - dropout_p);  // Inverted dropout scaling
        }
    }

    // Add weighted value with dropout-adjusted weight
    for (int d = 0; d < headdim; d++) {
        out_local[d] += exp_score_for_v * float(V[v_offset + d]);
    }

    max_score = new_max;
}

// Handle case where no keys are in range (empty attention)
if (sum_exp == 0.0f) {
    sum_exp = 1.0f;  // Avoid division by zero
    max_score = 0.0f;
}

// Normalize output and write
float norm_factor = 1.0f / sum_exp;
for (int d = 0; d < headdim; d++) {
    O[o_offset + d] = T(out_local[d] * norm_factor);
}

// Write logsumexp
L[lse_offset] = max_score + log(sum_exp);
