def peak_cpu_memory(param_sizes, strategy: str):
    if strategy == "baseline":
        return 2 * sum(param_sizes)
    if strategy == "sequential":
        return sum(param_sizes)
    if strategy == "meta_sequential":
        return max(param_sizes, default=0)

    raise ValueError(f"Invalid loading strategy: {strategy}")