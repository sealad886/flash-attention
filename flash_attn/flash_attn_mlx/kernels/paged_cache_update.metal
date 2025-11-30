/**
 * Paged KV Cache Update Kernel
 *
 * Writes newly computed key/value vectors into a paged KV cache using the
 * provided page table and per-sequence cache offsets. Each thread copies a
 * single (batch, token, head, dim) element from the new tensors into the
 * appropriate physical page based on cache_seqlens.
 */

uint elem = thread_position_in_grid.x;

if (batch_size <= 0 || seqlen_new <= 0 || nheads_k <= 0 || headdim <= 0) {
    return;
}

int total_elements = batch_size * seqlen_new * nheads_k * headdim;
if (elem >= uint(total_elements)) {
    return;
}

int dim = int(elem % uint(headdim));
int tmp = int(elem / uint(headdim));
int head = tmp % nheads_k;
tmp /= nheads_k;
int token_idx = tmp % seqlen_new;
int batch = tmp / seqlen_new;

if (batch < 0 || batch >= batch_size) {
    return;
}

int safe_page_size = (page_size > 0) ? page_size : 1;
int max_token_capacity = safe_page_size * max_pages_per_seq;
if (max_token_capacity <= 0) {
    return;
}

int start_pos = cache_seqlens[batch];
if (start_pos < 0) {
    start_pos = 0;
}

int target_pos = start_pos + token_idx;
if (target_pos < 0 || target_pos >= max_token_capacity) {
    return;
}

int log2_page = log2_page_size;
int page_mask = (log2_page >= 0) ? ((1 << log2_page) - 1) : 0;

int page_idx;
int page_offset;
if (log2_page >= 0) {
    page_idx = target_pos >> log2_page;
    page_offset = target_pos & page_mask;
} else {
    page_idx = target_pos / safe_page_size;
    page_offset = target_pos % safe_page_size;
}

if (page_idx < 0 || page_idx >= max_pages_per_seq) {
    return;
}

int table_index = batch * max_pages_per_seq + page_idx;
int physical_page = page_table[table_index];
if (physical_page < 0 || physical_page >= num_pages) {
    return;
}

int row_stride = nheads_k * headdim;
if (row_stride <= 0) {
    return;
}
int page_stride = safe_page_size * row_stride;

int cache_offset = physical_page * page_stride
                 + page_offset * row_stride
                 + head * headdim
                 + dim;

int batch_stride = seqlen_new * row_stride;
int new_offset = batch * batch_stride
               + token_idx * row_stride
               + head * headdim
               + dim;

k_cache[cache_offset] = new_k[new_offset];
v_cache[cache_offset] = new_v[new_offset];
