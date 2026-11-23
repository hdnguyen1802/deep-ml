def evaluate_preemption(
    num_tokens_list: list,
    kv_cache_bytes_per_token: int,
    swap_bandwidth_gbps: float,
    base_flops_per_token: int,
    attn_flops_per_token_pair: int,
    compute_tflops: float,
) -> dict:
    bandwidth_bytes_per_sec = swap_bandwidth_gbps * 10**9
    compute_flops_per_sec = compute_tflops * 10**12

    strategies = []
    swap_costs_ms = []
    recompute_costs_ms = []
    total_cost_ms = 0.0

    for n in num_tokens_list:
        swap_bytes = 2 * n * kv_cache_bytes_per_token
        swap_ms = swap_bytes / bandwidth_bytes_per_sec * 1000

        recompute_flops = n * base_flops_per_token + n * n * attn_flops_per_token_pair
        recompute_ms = recompute_flops / compute_flops_per_sec * 1000

        rounded_swap = round(swap_ms, 4)
        rounded_recompute = round(recompute_ms, 4)
        swap_costs_ms.append(rounded_swap)
        recompute_costs_ms.append(rounded_recompute)

        if swap_ms <= recompute_ms:
            strategies.append("swap")
            total_cost_ms += rounded_swap
        else:
            strategies.append("recompute")
            total_cost_ms += rounded_recompute

    return {
        "strategies": strategies,
        "swap_costs_ms": swap_costs_ms,
        "recompute_costs_ms": recompute_costs_ms,
        "total_cost_ms": round(total_cost_ms, 4),
    }