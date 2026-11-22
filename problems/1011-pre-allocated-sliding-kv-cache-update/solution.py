import numpy as np


def sliding_kv_cache_update(
    cache_k, cache_v, current_len, new_k, new_v, window_size
):
    new_k = np.asarray(new_k)
    new_v = np.asarray(new_v)
    new_len = new_k.shape[2]

    updated_len = min(current_len + new_len, window_size)
    keep_new = min(new_len, window_size)
    keep_old = updated_len - keep_new

    def update_cache(cache, new_entries):
        original = np.asarray(cache)
        updated = original.astype(float, copy=True)

        if 0 < keep_old < current_len:
            old_start = current_len - keep_old
            updated[:, :, :keep_old] = original[:, :, old_start:current_len]

        if keep_new:
            new_start = new_len - keep_new
            updated[:, :, keep_old:updated_len] = new_entries[:, :, new_start:]

        return updated.tolist()

    query_positions = np.maximum(updated_len - new_len + np.arange(new_len), 0)
    key_positions = np.arange(updated_len)
    mask = (key_positions <= query_positions[:, None]).astype(int)

    return {
        "cache_k": update_cache(cache_k, new_k),
        "cache_v": update_cache(cache_v, new_v),
        "current_len": updated_len,
        "mask": mask.tolist(),
    }