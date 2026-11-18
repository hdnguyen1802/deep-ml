def classify_llm_phases(
    num_params: int,
    sequence_length: int,
    batch_size: int,
    bytes_per_param: int,
    peak_flops: float,
    peak_bandwidth: float,
) -> dict:
    memory_bytes = num_params * bytes_per_param
    ridge_point = peak_flops / peak_bandwidth

    def analyze_phase(token_count: int) -> dict:
        total_flops = 2 * num_params * token_count
        arithmetic_intensity = total_flops / memory_bytes

        if arithmetic_intensity >= ridge_point:
            bottleneck = "compute-bound"
            achieved_flops = peak_flops
        else:
            bottleneck = "memory-bound"
            achieved_flops = arithmetic_intensity * peak_bandwidth

        utilization_percent = 100 * achieved_flops / peak_flops

        return {
            "total_flops": total_flops,
            "memory_bytes": memory_bytes,
            "arithmetic_intensity": round(arithmetic_intensity, 4),
            "bottleneck": bottleneck,
            "achieved_flops": round(achieved_flops, 4),
            "utilization_percent": round(utilization_percent, 4),
        }

    return {
        "ridge_point": round(ridge_point, 4),
        "prefill": analyze_phase(sequence_length),
        "decode": analyze_phase(batch_size),
    }