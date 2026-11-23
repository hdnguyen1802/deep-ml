import numpy as np


def paged_attention(query, key_cache, value_cache, block_table, context_len):
    if context_len == 0:
        return np.zeros(query.shape, dtype=float)

    block_size = key_cache.shape[1]
    positions = np.arange(context_len)
    logical_blocks, slots = np.divmod(positions, block_size)
    physical_blocks = np.asarray(block_table, dtype=np.intp)[logical_blocks]

    keys = key_cache[physical_blocks, slots]
    values = value_cache[physical_blocks, slots]

    head_dim = query.shape[-1]
    scores = np.einsum("hd,thd->ht", query, keys, dtype=float) / np.sqrt(head_dim)

    exp_scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

    output = np.einsum("ht,thd->hd", weights, values)
    return np.round(output, 4)