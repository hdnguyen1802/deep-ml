import torch

def calculate_latency_percentiles(latencies: torch.Tensor) -> dict[str, float]:
    if latencies.numel() == 0:
        return {'P50': 0.0, 'P95': 0.0, 'P99': 0.0}

    percentiles = torch.tensor([0.50, 0.95, 0.99], dtype=torch.float64, device=latencies.device)
    quantiles = torch.quantile(latencies, percentiles)

    return {
        'P50': round(quantiles[0].item(), 4),
        'P95': round(quantiles[1].item(), 4),
        'P99': round(quantiles[2].item(), 4),
    }