/**
 * Flash Attention Backward Kernel - dK/dV computation
 *
 * This kernel computes the gradients with respect to keys and values (dK, dV).
 * Uses the split backward design to avoid FP32 atomics.
 *
 * Grid: (batch * nheads_k, seqlen_k, 1)
 * Each thread computes dK and dV for one (batch, kv_head, key_pos) tuple.
 * For GQA/MQA, accumulates gradients across all Q heads sharing this KV head.
 *
 * Input layout: [batch, seqlen, nheads, headdim]
 * LSE layout: [batch, nheads, seqlen]
 *
 * Features:
 * - Causal masking: only attend to positions <= current position
 * - Sliding window: only attend to positions within (query_pos - window_left, query_pos + window_right)
 * - GQA/MQA: accumulates gradients across all Q heads sharing this KV head
 * - ALiBi: position-based attention bias with per-head slopes
 * - Dropout: regenerates Philox dropout mask for gradient computation
 *
 * Algorithm:
 *   For each key position k:
 *     For each query head h sharing this KV head:
 *       For each query position q (within window containing k):
 *         D = sum(dO[q] * O[q])
 *         P[q,k] = exp(score[q,k] + alibi_bias - L[q])
 *         If dropout: apply same dropout mask (via Philox RNG) and scale
 *         dS[q,k] = P[q,k] * (dO[q] @ V[k] - D) * scale
 *         dK[k] += dS[q,k] * Q[q]
 *         dV[k] += P[q,k] * dO[q]
 */

// Thread and grid indices
uint batch_head = thread_position_in_grid.x;
uint kv_block = thread_position_in_grid.y;

// Unpack scalar inputs
int batch = batch_size;
int sq = seq_len_q;
int sk = seq_len_k;
int nh = num_heads;
int nhk = num_heads_k;
int hd = head_dim;
bool is_causal = (causal_flag != 0);
bool apply_alibi = (use_alibi != 0);

uint batch_idx = batch_head / uint(nhk);
uint head_idx_k = batch_head % uint(nhk);
uint head_ratio = uint(nh) / uint(nhk);

if (batch_idx >= uint(batch)) return;

// Compute key/value position
uint kpos = kv_block;
if (kpos >= uint(sk)) return;

// Index strides
uint q_batch_stride = uint(sq) * uint(nh) * uint(hd);
uint k_batch_stride = uint(sk) * uint(nhk) * uint(hd);

uint k_offset = batch_idx * k_batch_stride + kpos * uint(nhk) * uint(hd) + head_idx_k * uint(hd);
uint dk_offset = k_offset;
uint dv_offset = k_offset;

// Load K row and V row
float k_row[256];
float v_row[256];
for (uint d = 0; d < uint(hd); d++) {
    k_row[d] = float(K[k_offset + d]);
    v_row[d] = float(V[k_offset + d]);
}

// Initialize dK and dV accumulators - accumulate across all Q heads that share this KV head
float dk_acc[256];
float dv_acc[256];
for (uint d = 0; d < uint(hd); d++) {
    dk_acc[d] = 0.0f;
    dv_acc[d] = 0.0f;
}

// Compute query range that can attend to this key position
// This key position k is visible to query position q if:
//   (q - window_left <= k) AND (k <= q + window_right) AND (is_causal ? k <= q : true)
// Rearranging:
//   q >= k - window_right (if window_right >= 0)
//   q <= k + window_left (if window_left >= 0)
//   q >= k (if causal)
int q_start = 0;
if (is_causal) {
    q_start = max(q_start, int(kpos));  // Query must be >= key for causal
}
if (window_right >= 0) {
    q_start = max(q_start, int(kpos) - window_right);  // key_pos - window_right <= query_pos
}

int q_end = sq;
if (window_left >= 0) {
    q_end = min(q_end, int(kpos) + window_left + 1);  // query_pos <= key_pos + window_left
}

// Loop over Q heads that share this KV head
for (uint h = 0; h < head_ratio; h++) {
    uint head_idx = head_idx_k * head_ratio + h;

    // Compute ALiBi slope for this head
    float alibi_slope = 0.0f;
    if (apply_alibi) {
        int slope_idx = (alibi_batch_stride > 0) ? int(batch_idx) * alibi_batch_stride + int(head_idx) : int(head_idx);
        alibi_slope = alibi_slopes[slope_idx];
    }

    // Loop over query positions that can attend to this key
    for (int qpos = q_start; qpos < q_end; qpos++) {
        // Compute Q/O/dO offset
        uint q_offset = batch_idx * q_batch_stride + uint(qpos) * uint(nh) * uint(hd) + head_idx * uint(hd);
        uint l_offset = batch_idx * uint(nh) * uint(sq) + head_idx * uint(sq) + uint(qpos);

        // Load Q row, O row, dO row
        float q_row[256];
        float o_row[256];
        float do_row[256];
        for (uint d = 0; d < uint(hd); d++) {
            q_row[d] = float(Q[q_offset + d]);
            o_row[d] = float(O[q_offset + d]);
            do_row[d] = float(dO[q_offset + d]);
        }
        float L_val = L[l_offset];

        // Compute D = rowsum(dO * O)
        float D = 0.0f;
        for (uint d = 0; d < uint(hd); d++) {
            D += do_row[d] * o_row[d];
        }

        // Compute score = Q @ K^T
        float score = 0.0f;
        for (uint d = 0; d < uint(hd); d++) {
            score += q_row[d] * k_row[d];
        }
        score *= softmax_scale;

        // Add ALiBi bias if enabled
        if (apply_alibi) {
            int distance = is_causal ? (int(kpos) - qpos) : -abs(qpos - int(kpos));
            score += alibi_slope * float(distance);
        }

        // Compute P = softmax(score) using saved L
        float P = exp(score - L_val);

        // Apply dropout if enabled (regenerate same mask as forward pass)
        float P_dropped = P;
        if (dropout_p > 0.0f) {
            // Philox 4x32-7 RNG - same counter/key as forward pass
            // Counter: (query_pos, key_pos, batch, head)
            // Key: (philox_seed, philox_offset)
            uint4 ctr = uint4(uint(qpos), uint(kpos), uint(batch_idx), uint(head_idx));
            uint2 key = uint2(uint(philox_seed), uint(philox_offset));

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

            // Convert to float in [0, 1)
            float rand_val = float(ctr.x) * 2.3283064365386963e-10f;

            if (rand_val < dropout_p) {
                P_dropped = 0.0f;  // This position was dropped in forward
            } else {
                P_dropped = P / (1.0f - dropout_p);  // Inverted dropout scaling
            }
        }

        // Compute dS = P_dropped * (dO @ V^T - D)
        float doV = 0.0f;
        for (uint d = 0; d < uint(hd); d++) {
            doV += do_row[d] * v_row[d];
        }
        float dS = P_dropped * (doV - D) * softmax_scale;

        // Accumulate dK = dS^T @ Q
        for (uint d = 0; d < uint(hd); d++) {
            dk_acc[d] += dS * q_row[d];
        }

        // Accumulate dV = P_dropped^T @ dO
        for (uint d = 0; d < uint(hd); d++) {
            dv_acc[d] += P_dropped * do_row[d];
        }
    }
}

// Write dK and dV
for (uint d = 0; d < uint(hd); d++) {
    dK[dk_offset + d] = T(dk_acc[d]);
    dV[dv_offset + d] = T(dv_acc[d]);
}
