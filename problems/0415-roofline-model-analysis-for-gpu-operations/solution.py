def roofline_analysis(
    peak_gflops: float,
    peak_bandwidth_gbs: float,
    operations: list,
) -> dict:
    ridge_point = peak_gflops / peak_bandwidth_gbs
    results = []

    for operation in operations:
        intensity = operation["flops"] / operation["bytes"]

        attainable_gflops = min(
            peak_gflops,
            intensity * peak_bandwidth_gbs,
        )

        bottleneck = (
            "memory-bound"
            if intensity < ridge_point
            else "compute-bound"
        )

        efficiency = attainable_gflops / peak_gflops * 100

        results.append({
            "name": operation["name"],
            "operational_intensity": intensity,
            "attainable_gflops": attainable_gflops,
            "bottleneck": bottleneck,
            "efficiency": efficiency,
        })

    return {
        "ridge_point": ridge_point,
        "operations": results,
    }