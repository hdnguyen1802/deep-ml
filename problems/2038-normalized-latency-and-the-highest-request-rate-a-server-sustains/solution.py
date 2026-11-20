def _mean_normalized_latency(requests):
    total = sum(
        (finish - arrival) / output_len
        for arrival, finish, output_len in requests
    )
    return total / len(requests)


def normalized_latency(requests):
    return round(_mean_normalized_latency(requests), 4)


def max_sustainable_rate(traces, threshold):
    qualifying_rates = (
        rate
        for rate, requests in traces.items()
        if _mean_normalized_latency(requests) <= threshold
    )
    return max(qualifying_rates, default=None)


def capacity_gain(traces_a, traces_b, threshold):
    rate_a = max_sustainable_rate(traces_a, threshold)
    rate_b = max_sustainable_rate(traces_b, threshold)

    if rate_a is None or rate_b is None:
        return None

    return round(rate_a / rate_b, 2)