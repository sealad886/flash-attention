/**
 * Shared Utilities for Flash Attention Metal Kernels
 *
 * This file contains common utilities, helper functions, and constants
 * used across all Flash Attention kernels.
 */

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;


// ============================================================================
// Constants
// ============================================================================

// Numerical constants
constant float NEG_INFINITY = -INFINITY;
constant float LOG2E = 1.4426950408889634f;  // log2(e)


// ============================================================================
// Math Utilities
// ============================================================================

/**
 * Compute exp2 (2^x) - faster than exp on Metal
 */
inline float fast_exp(float x) {
    return exp2(x * LOG2E);
}

/**
 * Compute log2
 */
inline float fast_log(float x) {
    return log2(x) / LOG2E;
}

/**
 * Compute safe log (avoids log(0))
 */
inline float safe_log(float x) {
    return log(max(x, 1e-20f));
}

/**
 * Compute tanh for softcap
 */
inline float fast_tanh(float x) {
    // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    float e2x = exp2(2.0f * x * LOG2E);
    return (e2x - 1.0f) / (e2x + 1.0f);
}


// ============================================================================
// Mask Utilities
// ============================================================================

/**
 * Check if position should be masked (causal attention)
 *
 * @param query_pos Query position
 * @param key_pos Key position
 * @return true if key_pos > query_pos (should be masked)
 */
inline bool is_causal_masked(int query_pos, int key_pos) {
    return key_pos > query_pos;
}

/**
 * Check if position should be masked (local/sliding window attention)
 *
 * @param query_pos Query position
 * @param key_pos Key position
 * @param window_left Left window size (-1 = no limit)
 * @param window_right Right window size (-1 = no limit)
 * @return true if position is outside window
 */
inline bool is_window_masked(int query_pos, int key_pos, int window_left, int window_right) {
    if (window_left >= 0 && key_pos < query_pos - window_left) {
        return true;
    }
    if (window_right >= 0 && key_pos > query_pos + window_right) {
        return true;
    }
    return false;
}

/**
 * Combined mask check (causal + window)
 */
inline bool is_masked(int query_pos, int key_pos, bool causal, int window_left, int window_right) {
    if (causal && is_causal_masked(query_pos, key_pos)) {
        return true;
    }
    return is_window_masked(query_pos, key_pos, window_left, window_right);
}


// ============================================================================
// ALiBi Utilities
// ============================================================================

/**
 * Compute ALiBi bias for a single position
 *
 * @param query_pos Query position
 * @param key_pos Key position
 * @param slope ALiBi slope for this head
 * @param causal Whether using causal attention
 * @return ALiBi bias value
 */
inline float compute_alibi_bias(int query_pos, int key_pos, float slope, bool causal) {
    int distance;
    if (causal) {
        // For causal: bias = slope * (key_pos - query_pos), which is <= 0
        distance = key_pos - query_pos;
    } else {
        // For bidirectional: bias = -slope * |query_pos - key_pos|
        distance = -abs(query_pos - key_pos);
    }
    return slope * float(distance);
}


// ============================================================================
// Philox RNG Utilities
// ============================================================================

inline float philox_uniform(
    int query_pos,
    int key_pos,
    int batch_idx,
    int head_idx,
    uint philox_seed,
    uint philox_offset
) {
    uint4 ctr = uint4(uint(query_pos), uint(key_pos), uint(batch_idx), uint(head_idx));
    uint2 key = uint2(philox_seed, philox_offset);

    for (int round = 0; round < 7; ++round) {
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

    return float(ctr.x) * 2.3283064365386963e-10f;
}


// ============================================================================
// Softmax Utilities
// ============================================================================

/**
 * Online softmax state for a single row
 */
struct SoftmaxState {
    float max_val;  // Running maximum
    float sum_exp;  // Running sum of exp(x - max)

    // Initialize state
    inline void init() {
        max_val = NEG_INFINITY;
        sum_exp = 0.0f;
    }

    // Update state with a new value
    inline void update(float val) {
        if (val > max_val) {
            sum_exp = sum_exp * exp2((max_val - val) * LOG2E) + 1.0f;
            max_val = val;
        } else {
            sum_exp += exp2((val - max_val) * LOG2E);
        }
    }

    // Merge another state into this one
    inline void merge(SoftmaxState other) {
        if (other.max_val > max_val) {
            sum_exp = sum_exp * exp2((max_val - other.max_val) * LOG2E) + other.sum_exp;
            max_val = other.max_val;
        } else if (max_val > other.max_val) {
            sum_exp += other.sum_exp * exp2((other.max_val - max_val) * LOG2E);
        } else {
            sum_exp += other.sum_exp;
        }
    }

    // Get log-sum-exp
    inline float logsumexp() {
        return max_val + log2(sum_exp) / LOG2E;
    }

    // Get normalization factor (1 / sum_exp)
    inline float norm_factor() {
        return 1.0f / sum_exp;
    }
};


// ============================================================================
// Threadgroup Reduction Utilities
// ============================================================================

/**
 * Reduce maximum across threadgroup using simdgroup operations
 *
 * @param val Value to reduce
 * @param tg_mem Threadgroup memory for reduction
 * @param tid Thread index in threadgroup
 * @return Maximum value across all threads
 */
inline float threadgroup_max(float val, threadgroup float* tg_mem, uint tid, uint tg_size) {
    const uint SIMD_WIDTH = 32;

    // First reduce within SIMD group
    val = simd_max(val);

    // Store to threadgroup memory from the first lane in each SIMD group
    if ((tid & (SIMD_WIDTH - 1)) == 0) {
        uint simd_group = tid / SIMD_WIDTH;
        tg_mem[simd_group] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction by first thread of the threadgroup
    if (tid == 0) {
        uint num_simd_groups = (tg_size + (SIMD_WIDTH - 1)) / SIMD_WIDTH;
        for (uint i = 1; i < num_simd_groups; i++) {
            val = max(val, tg_mem[i]);
        }
        tg_mem[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    return tg_mem[0];
}

/**
 * Reduce sum across threadgroup using simdgroup operations
 */
inline float threadgroup_sum(float val, threadgroup float* tg_mem, uint tid, uint tg_size) {
    const uint SIMD_WIDTH = 32;

    // First reduce within SIMD group
    val = simd_sum(val);

    // Store to threadgroup memory from the first lane in each SIMD group
    if ((tid & (SIMD_WIDTH - 1)) == 0) {
        uint simd_group = tid / SIMD_WIDTH;
        tg_mem[simd_group] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction by first thread of the threadgroup
    if (tid == 0) {
        uint num_simd_groups = (tg_size + (SIMD_WIDTH - 1)) / SIMD_WIDTH;
        for (uint i = 1; i < num_simd_groups; i++) {
            val += tg_mem[i];
        }
        tg_mem[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    return tg_mem[0];
}


// ============================================================================
// Data Type Conversion
// ============================================================================

/**
 * Convert half to float
 */
inline float to_float(half x) {
    return float(x);
}

/**
 * Convert bfloat16 to float
 */
inline float to_float(bfloat x) {
    return float(x);
}

/**
 * Convert float to half
 */
inline half to_half(float x) {
    return half(x);
}

/**
 * Convert float to bfloat16
 */
inline bfloat to_bfloat(float x) {
    return bfloat(x);
}
