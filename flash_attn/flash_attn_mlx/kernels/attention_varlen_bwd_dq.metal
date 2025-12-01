    uint elem = thread_position_in_grid.x;
    long total_threads = long(total_q) * long(nheads);
    if (elem >= total_threads) {
        return;
    }

    if (batch_size <= 0) {
        return;
    }

    bool apply_alibi = (use_alibi != 0);
    bool is_causal = (causal_flag != 0);

    int head_idx = int(elem % uint(nheads));
    int global_q_idx = int(elem / uint(nheads));

    if (global_q_idx >= total_q) {
        return;
    }

    int batch_idx = 0;
    {
        int lo = 0;
        int hi = batch_size;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            int mid_end = cu_seqlens_q[mid + 1];
            if (global_q_idx >= mid_end) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        batch_idx = min(lo, batch_size - 1);
    }
    if (batch_idx < 0) {
        return;
    }

    int q_seq_start = cu_seqlens_q[batch_idx];
    int q_seq_end = cu_seqlens_q[batch_idx + 1];
    int seq_len_q = q_seq_end - q_seq_start;
    if (seq_len_q <= 0) {
        return;
    }

    int local_q_pos = global_q_idx - q_seq_start;
    if (local_q_pos < 0 || local_q_pos >= seq_len_q) {
        return;
    }

    int k_seq_start = cu_seqlens_k[batch_idx];
    int k_seq_end = cu_seqlens_k[batch_idx + 1];
    int seq_len_k = k_seq_end - k_seq_start;
    if (seq_len_k <= 0) {
        int dq_offset_zero = (global_q_idx * nheads + head_idx) * headdim;
        for (int d = 0; d < headdim; ++d) {
            dQ[dq_offset_zero + d] = T(0.0f);
        }
        return;
    }

    int kv_ratio = max(1, nheads / max(1, nheads_k));
    int kv_head = head_idx / kv_ratio;
    if (kv_head >= nheads_k) {
        kv_head = nheads_k - 1;
    }

    // Compute ALiBi slope for this head
    float alibi_slope = 0.0f;
    if (apply_alibi) {
        int slope_idx = (alibi_batch_stride > 0) ? batch_idx * alibi_batch_stride + head_idx : head_idx;
        alibi_slope = alibi_slopes[slope_idx];
    }

    int q_offset = (global_q_idx * nheads + head_idx) * headdim;
    int dq_offset = q_offset;
    int o_offset = q_offset;
    int do_offset = q_offset;
    int l_offset = head_idx * total_q + global_q_idx;

    float q_row[256];
    float o_row[256];
    float do_row[256];
    for (int d = 0; d < headdim; ++d) {
        q_row[d] = float(Q[q_offset + d]);
        o_row[d] = float(O[o_offset + d]);
        do_row[d] = float(dO[do_offset + d]);
    }

    float L_val = L[l_offset];
    float D = 0.0f;
    for (int d = 0; d < headdim; ++d) {
        D += do_row[d] * o_row[d];
    }

    float dq_acc[256];
    for (int d = 0; d < headdim; ++d) {
        dq_acc[d] = 0.0f;
    }

    float k_row[256];
    float v_row[256];

    int k_begin = 0;
    int k_end = seq_len_k;
    if (causal_flag != 0) {
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

    float scale = softmax_scale;

    for (int local_k = k_begin; local_k < k_end; ++local_k) {
        int global_k_idx = k_seq_start + local_k;
        int k_offset = (global_k_idx * nheads_k + kv_head) * headdim;

        for (int d = 0; d < headdim; ++d) {
            k_row[d] = float(K[k_offset + d]);
            v_row[d] = float(V[k_offset + d]);
        }

        float score = 0.0f;
        for (int d = 0; d < headdim; ++d) {
            score += q_row[d] * k_row[d];
        }
        score *= scale;

        // Add ALiBi bias if enabled (uses local positions within segment)
        if (apply_alibi) {
            int distance = is_causal ? (local_k - local_q_pos) : -abs(local_q_pos - local_k);
            score += alibi_slope * float(distance);
        }

        if (softcap > 0.0f) {
            score = softcap * tanh(score / softcap);
        }

        float P = exp(score - L_val);

        float P_dropped = P;
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
                P_dropped = 0.0f;
            } else {
                P_dropped = P / (1.0f - dropout_p);
            }
        }

        float doV = 0.0f;
        for (int d = 0; d < headdim; ++d) {
            doV += do_row[d] * v_row[d];
        }

        float dS = P_dropped * (doV - D) * scale;

        for (int d = 0; d < headdim; ++d) {
            dq_acc[d] += dS * k_row[d];
        }
    }

    for (int d = 0; d < headdim; ++d) {
        dQ[dq_offset + d] = T(dq_acc[d]);
    }
