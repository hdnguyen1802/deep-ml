import numpy as np


def kv_cache_attention_step(x_new, W_Q, W_K, W_V, cache):
    q = x_new @ W_Q
    k = x_new @ W_K
    v = x_new @ W_V

    if cache is None:
        K_cache = k[None, :]
        V_cache = v[None, :]
    else:
        K_cache, V_cache = cache
        K_cache = np.vstack((K_cache, k))
        V_cache = np.vstack((V_cache, v))

    d_k = k.shape[0]
    scores = (q @ K_cache.T) / np.sqrt(d_k)

    exp_scores = np.exp(scores - np.max(scores))
    weights = exp_scores / np.sum(exp_scores)

    output = weights @ V_cache
    return output, (K_cache, V_cache)