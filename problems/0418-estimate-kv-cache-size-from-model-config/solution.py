def estimate_kv_cache_size(
    model_config: dict, batch_size: int, seq_len: int
) -> dict:
    num_layers = model_config["num_layers"]
    num_attention_heads = model_config["num_attention_heads"]
    num_kv_heads = model_config.get("num_kv_heads", num_attention_heads)
    head_dim = model_config["hidden_size"] // num_attention_heads
    dtype_bytes = model_config["dtype_bytes"]

    elements_per_token = 2 * num_layers * num_kv_heads * head_dim
    bytes_per_token = elements_per_token * dtype_bytes

    kv_cache_elements = elements_per_token * seq_len * batch_size
    kv_cache_size_bytes = kv_cache_elements * dtype_bytes
    kv_cache_size_mb = kv_cache_size_bytes / (1024 * 1024)

    return {
        "kv_cache_elements": kv_cache_elements,
        "kv_cache_size_bytes": kv_cache_size_bytes,
        "kv_cache_size_mb": round(kv_cache_size_mb, 4),
        "per_layer_size_mb": round(kv_cache_size_mb / num_layers, 4),
        "per_token_size_kb": round(bytes_per_token / 1024, 4),
    }