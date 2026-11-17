import torch

def decompose_latency(stage_latencies: dict, percentiles: list) -> dict:
    """
    Decompose end-to-end inference latency into component stages.
    
    Args:
        stage_latencies: dict mapping stage name -> torch.Tensor of latency measurements (ms)
        percentiles: list of percentile values to compute (e.g., [50, 95, 99])
    
    Returns:
        Dictionary with keys: 'e2e_mean', 'e2e_percentiles', 'stage_stats',
                               'bottleneck', 'stage_pct'
    """
    result = {}
    e2e = torch.stack(list(stage_latencies.values())).sum(dim=0)
    result['e2e_mean'] = round(e2e.mean().item(),2)
    p_e2e = torch.tensor([p/100 for p in percentiles], dtype = e2e.dtype)
    quantile_values = torch.quantile(e2e,p_e2e)
    result['e2e_percentiles'] = {p:round(q.item(),2) for p,q in zip(percentiles,quantile_values)}
    stage_stats = {}
    stage_means = {}
    for name,val in stage_latencies.items():
        quantile = torch.quantile(val,p_e2e)
        s_mean = round(val.mean().item(),2)
        stage_means[name] = s_mean
        stage_stats[name] = {
            'mean' : s_mean,
            'std' : round(val.std(correction=0).item(),2),
            'percentiles' : {p:round(q.item(),2) for p,q in zip(percentiles,quantile)}
        }
    result['stage_stats'] = stage_stats
    result['bottleneck'] = max(stage_means, key=stage_means.get)
    result['stage_pct'] = {
        name : round((s_mean/result['e2e_mean']*100),2) for name,s_mean in stage_means.items()
    }

    return result
