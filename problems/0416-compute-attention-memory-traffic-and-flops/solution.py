def attention_memory_flops(
    B: int, h: int, N: int, d: int, bytes_per_element: int
) -> dict:
    tensor_elements = B * h * N * d
    score_elements = B * h * N * N

    qk_flops = 2 * d * score_elements
    softmax_flops = 5 * score_elements
    pv_flops = 2 * N * tensor_elements
    total_flops = qk_flops + softmax_flops + pv_flops

    # Read Q, K, and V once. Write O once.
    # Write and read S and P once each.
    memory_bytes = (4 * tensor_elements + 4 * score_elements) * bytes_per_element

    return {
        "qk_flops": qk_flops,
        "softmax_flops": softmax_flops,
        "pv_flops": pv_flops,
        "total_flops": total_flops,
        "memory_bytes": memory_bytes,
        "arithmetic_intensity": round(total_flops / memory_bytes, 2),
    }