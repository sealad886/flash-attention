/**
 * Flash Attention Backward Kernel - dQ computation
 *
 * This kernel computes the gradient with respect to queries (dQ).
 * Uses the split backward design to avoid FP32 atomics.
 *
 * Grid: (batch * nheads, seqlen_q, 1)
 * Each thread computes dQ for one (batch, head, query_pos) tuple.
 *
 * Input layout: [batch, seqlen, nheads, headdim]
 * LSE layout: [batch, nheads, seqlen]
 *
 * Features:
 * - Causal masking: only attend to positions <= current position
 * - Sliding window: only attend to positions within (query_pos - window_left, query_pos + window_right)
 * - GQA/MQA: nheads_k can be different from nheads
 * - ALiBi: position-based attention bias with per-head slopes
 * - Dropout: regenerates Philox dropout mask for gradient computation
 *
 * Algorithm:
 *   For each query position q:
 *     D = sum(dO[q] * O[q])  // Row-wise dot product
 *     For each key position k (within window):
 *       P[q,k] = exp(score[q,k] + alibi_bias - L[q])  // Reconstruct softmax
 *       If dropout: apply same dropout mask (via Philox RNG) and scale
 *       dS[q,k] = P[q,k] * (dO[q] @ V[k] - D) * scale
 *       dQ[q] += dS[q,k] * K[k]
 */

// Thread and grid indices
uint batch_head = thread_position_in_grid.x;
uint query_block = thread_position_in_grid.y;

// Unpack scalar inputs
int batch = batch_size;
int sq = seq_len_q;
int sk = seq_len_k;
int nh = num_heads;
int nhk = num_heads_k;
int hd = head_dim;
bool is_causal = (causal_flag != 0);
bool apply_alibi = (use_alibi != 0);

uint batch_idx = batch_head / uint(nh);
uint head_idx = batch_head % uint(nh);
uint head_idx_k = head_idx / (uint(nh) / uint(nhk));

if (batch_idx >= uint(batch)) return;

// Compute ALiBi slope for this head
float alibi_slope = 0.0f;
if (apply_alibi) {
    int slope_idx = (alibi_batch_stride > 0) ? int(batch_idx) * alibi_batch_stride + int(head_idx) : int(head_idx);
    alibi_slope = alibi_slopes[slope_idx];
}

if (batch_idx >= uint(batch)) return;

// Compute query position
uint qpos = query_block;
if (qpos >= uint(sq)) return;

// Index offsets - note: [batch, seqlen, nheads, headdim] layout
uint q_batch_stride = uint(sq) * uint(nh) * uint(hd);
uint k_batch_stride = uint(sk) * uint(nhk) * uint(hd);

uint q_offset = batch_idx * q_batch_stride + qpos * uint(nh) * uint(hd) + head_idx * uint(hd);
uint dq_offset = q_offset;  // Same layout
uint o_offset = q_offset;
uint do_offset = q_offset;
uint l_offset = batch_idx * uint(nh) * uint(sq) + head_idx * uint(sq) + qpos;

// Load Q row, O row, dO row, and L value
float q_row[256];  // Max head dim
float o_row[256];
float do_row[256];
for (uint d = 0; d < uint(hd); d++) {
    q_row[d] = float(Q[q_offset + d]);
    o_row[d] = float(O[o_offset + d]);
    do_row[d] = float(dO[do_offset + d]);
}
float L_val = L[l_offset];

// Compute D = rowsum(dO * O)
float D = 0.0f;
for (uint d = 0; d < uint(hd); d++) {
    D += do_row[d] * o_row[d];
}

// Initialize dQ accumulator
float dq_acc[256];
for (uint d = 0; d < uint(hd); d++) {
    dq_acc[d] = 0.0f;
}

// Compute key range for iteration
// Start: max(0, query_pos - window_left) if window_left >= 0, else 0
int k_start = (window_left >= 0) ? max(0, int(qpos) - window_left) : 0;

// End: min(seqlen_k, query_pos + window_right + 1) if window_right >= 0
//      For causal, also limit to query_pos + 1
int k_end = sk;
if (is_causal) {
    k_end = min(k_end, int(qpos) + 1);
}
if (window_right >= 0) {
    k_end = min(k_end, int(qpos) + window_right + 1);
}

// Loop over key positions within window
for (int kpos = k_start; kpos < k_end; kpos++) {
    // Compute K offset
    uint k_offset = batch_idx * k_batch_stride + uint(kpos) * uint(nhk) * uint(hd) + head_idx_k * uint(hd);
    uint v_offset = k_offset;

    // Load K row and V row
    float k_row[256];
    float v_row[256];
    for (uint d = 0; d < uint(hd); d++) {
        k_row[d] = float(K[k_offset + d]);
        v_row[d] = float(V[v_offset + d]);
    }

    // Compute score = Q @ K^T
    float score = 0.0f;
    for (uint d = 0; d < uint(hd); d++) {
        score += q_row[d] * k_row[d];
    }
    score *= softmax_scale;

    // Add ALiBi bias if enabled
    if (apply_alibi) {
        int distance;
        if (is_causal) {
            distance = kpos - int(qpos);
        } else {
            distance = -abs(int(qpos) - kpos);
        }
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

    // Accumulate dQ = dS @ K
    for (uint d = 0; d < uint(hd); d++) {
        dq_acc[d] += dS * k_row[d];
    }
}

// Write dQ
for (uint d = 0; d < uint(hd); d++) {
    dQ[dq_offset + d] = T(dq_acc[d]);
}
