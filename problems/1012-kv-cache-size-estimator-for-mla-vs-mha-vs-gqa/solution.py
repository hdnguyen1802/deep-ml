def kv_cache_bytes(
    batch: int,
    seqlen: int,
    n_layers: int,
    n_heads: int,
    head_dim: int,
    n_kv_groups: int,
    latent_dim: int,
    attn_type: str,
    bytes_per_elem: int = 2,
) -> int:
    if attn_type == "mha":
        elements_per_token_per_layer = 2 * n_heads * head_dim
    elif attn_type == "gqa":
        elements_per_token_per_layer = 2 * n_kv_groups * head_dim
    elif attn_type == "mla":
        elements_per_token_per_layer = latent_dim
    else:
        raise ValueError(f"Unknown attention type: {attn_type}")

    return batch * seqlen * n_layers * elements_per_token_per_layer * bytes_per_elem