def compute_inference_metrics(timestamps: list[float]) -> dict:
    """
    Compute LLM inference performance metrics from token timestamps.
    
    Args:
        timestamps: List of floats where timestamps[0] is the request start time
                    and timestamps[1:] are the times when each output token was generated.
    
    Returns:
        Dictionary with keys 'ttft', 'tps', 'itl' containing the metric values.
    """
    metrics = {}
    metrics['ttft'] = timestamps[1] - timestamps[0]
    metrics['tps'] = (len(timestamps)-1)/(timestamps[-1] - timestamps[0])
    metrics['itl'] = 0 if len(timestamps)==2 else (timestamps[-1] - timestamps[1])/(len(timestamps)-2)

    return metrics