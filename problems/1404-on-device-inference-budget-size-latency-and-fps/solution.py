def on_device_budget(
    model_mb,
    latency_ms,
    memory_budget_mb,
    target_fps,
    bits=32,
    sparsity=0.0,
    distill_ratio=1.0,
    speedup=None,
):
    if speedup is None:
        speedup = {32: 1.0, 16: 1.6, 8: 2.5, 4: 3.0}

    size_mb = model_mb * distill_ratio * (1 - sparsity) * (bits / 32)
    estimated_latency_ms = latency_ms * distill_ratio / speedup[bits]
    fps = 1000 / estimated_latency_ms

    fits_memory = size_mb <= memory_budget_mb
    meets_fps = round(fps,1) >= target_fps

    return {
        "size_mb": round(size_mb, 3),
        "latency_ms": round(estimated_latency_ms, 3),
        "fps": round(fps, 1),
        "fits_memory": fits_memory,
        "meets_fps": meets_fps,
        "ok": fits_memory and meets_fps,
    }