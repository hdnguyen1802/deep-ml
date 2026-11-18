def gpu_ops_byte_ratio(gpu_specs: dict) -> dict:
    bandwidth_bytes_per_second = gpu_specs["memory_bandwidth_gbps"] * 10**9

    return {
        precision: round(tflops * 10**12 / bandwidth_bytes_per_second, 2)
        for precision, tflops in gpu_specs["compute_tflops"].items()
    }