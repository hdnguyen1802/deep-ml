import math


def estimate_min_gpus(
    num_params_billion: float,
    bytes_per_param: int,
    gpu_memory_gb: float,
    overhead_fraction: float,
) -> dict:
    model_memory_gb = num_params_billion * bytes_per_param
    total_memory_gb = model_memory_gb * (1 + overhead_fraction)
    min_gpus = math.ceil(total_memory_gb / gpu_memory_gb)

    return {
        "model_memory_gb": round(model_memory_gb, 2),
        "total_memory_gb": round(total_memory_gb, 2),
        "min_gpus": min_gpus,
    }