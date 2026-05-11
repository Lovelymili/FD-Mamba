import torch
import torch.nn.functional as F



def exists(val):
    return val is not None


def is_empty(t):
    return t.nelement() == 0


def expand_dim(t, dim, k):
    t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def ema_inplace(moving_avg, new, decay):
    if is_empty(moving_avg):
        moving_avg.data.copy_(new)
        return
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def batched_bincount(index, num_classes, dim=-1):
    shape = list(index.shape)
    shape[dim] = num_classes
    out = index.new_zeros(shape)
    out.scatter_add_(dim, index, torch.ones_like(index, dtype=index.dtype))
    return out


def dists_and_buckets(x, means):
    
    
    x_norm = F.normalize(x, dim=-1)
    means_norm = F.normalize(means, dim=-1)
    
    sims = torch.einsum('b n c, k c -> b n k', x_norm, means_norm)
    _, buckets = torch.max(sims, dim=-1)
    return sims, buckets


def center_iter(x, means, buckets=None):
    
    b, l, d, dtype, num_tokens = *x.shape, x.dtype, means.shape[0]
    if not exists(buckets):
        _, buckets = dists_and_buckets(x, means)

    bins = batched_bincount(buckets, num_tokens).sum(0, keepdim=True)
    zero_mask = bins.long() == 0
    means_ = buckets.new_zeros(b, num_tokens, d, dtype=dtype)
    means_.scatter_add_(-2, expand_dim(buckets, -1, d), x)
    means_ = F.normalize(means_.sum(0, keepdim=True), dim=-1).type(dtype)

    
    means = torch.where(zero_mask.unsqueeze(-1), means, means_)
    return means.squeeze(0)