import torch


class KVCache:
    def __init__(self):
        self.reset()

    def append(self, k, v):
        if self.k is not None:
            k = torch.cat((self.k, k), dim=2)
            v = torch.cat((self.v, v), dim=2)

        self.k, self.v = k, v
        return self.k, self.v

    def __len__(self):
        return 0 if self.k is None else self.k.shape[2]

    def reset(self):
        self.k = None
        self.v = None


def attend_with_cache(q, k_new, v_new, cache):
    k, v = cache.append(k_new, v_new)
    scale = q.shape[-1] ** 0.5
    scores = (q @ k.transpose(-2, -1)) / scale
    weights = torch.softmax(scores, dim=-1)
    return weights @ v