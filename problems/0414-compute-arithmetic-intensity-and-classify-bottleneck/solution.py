def compute_arithmetic_intensity(
    flops: float,
    bytes_accessed: float,
    peak_performance: float,
    peak_bandwidth: float,
) -> dict:
    arithmetic_intensity = flops / bytes_accessed
    ridge_point = peak_performance / peak_bandwidth

    if arithmetic_intensity < ridge_point:
        bottleneck = "memory-bound"
        achieved_performance = arithmetic_intensity * peak_bandwidth
    else:
        bottleneck = "compute-bound"
        achieved_performance = peak_performance

    utilization_percent = achieved_performance / peak_performance * 100

    return {
        "arithmetic_intensity": round(arithmetic_intensity, 4),
        "ridge_point": round(ridge_point, 4),
        "bottleneck": bottleneck,
        "achieved_performance": round(achieved_performance, 4),
        "utilization_percent": round(utilization_percent, 4),
    }