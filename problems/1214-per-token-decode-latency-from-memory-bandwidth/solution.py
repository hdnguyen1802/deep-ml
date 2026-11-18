def estimate_decode_latency(num_params, bytes_per_param, bandwidth_bytes_per_s):
    weight_bytes = num_params * bytes_per_param
    seconds_per_token = weight_bytes / bandwidth_bytes_per_s

    latency_ms = seconds_per_token * 1000
    tokens_per_sec = 1 / seconds_per_token

    return [latency_ms, tokens_per_sec]