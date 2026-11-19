def microbatch_speedup(
    batch_time: float,
    num_microbatches: int,
    overhead_per_microbatch: float,
) -> float:
    time_mb = (
        batch_time / num_microbatches
        + num_microbatches * overhead_per_microbatch
    )
    return batch_time / time_mb