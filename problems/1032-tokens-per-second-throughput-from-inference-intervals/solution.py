def compute_throughput(intervals):
    step_tps = []
    cumulative_tps = []
    total_tokens = 0
    total_seconds = 0

    for tokens, seconds in intervals:
        if seconds <= 0:
            raise ValueError("Elapsed seconds must be greater than zero.")
        if tokens < 0:
            raise ValueError("Tokens must not be negative.")

        total_tokens += tokens
        total_seconds += seconds

        step_tps.append(round(tokens / seconds, 2))
        cumulative_tps.append(round(total_tokens / total_seconds, 2))

    overall_tps = (
        round(total_tokens / total_seconds, 2)
        if total_seconds > 0
        else 0.0
    )

    return {
        "step_tps": step_tps,
        "cumulative_tps": cumulative_tps,
        "overall_tps": overall_tps,
    }