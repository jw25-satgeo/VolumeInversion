import torch
import torch.nn.functional as F

def nanpute_n1mean(x: torch.Tensor, max_iters=None, dims=(0,1)) -> torch.Tensor:
    '''
    Infill NaN values by averaging immediate neighbours. 
    '''
    if max_iters is None: max_iters = sum([x.shape[dim] for dim in dims])
    sl = tuple(slice(1, -1) if i in dims else slice(None) for i in range(x.ndim))
    __x = torch.full(tuple(x.shape[i]+2 if i in dims else x.shape[i] for i in range(x.ndim)), torch.nan, device=x.device, dtype=x.dtype)
    for _ in range(max_iters):
        __x[sl] = x; # pad before unfold to preserve spatial dimension
        nv, _x, _v = ~torch.isfinite(x), __x, torch.isfinite(__x)
        if not nv.any(): break
        for dim in dims: _x, _v = _x.unfold(dim,3,1), _v.unfold(dim,3,1) # restores padded dimension to orig
        x[nv] = torch.where(_v[nv], _x[nv], 0).flatten(-len(dims)).sum(-1) / _v[nv].flatten(-len(dims)).sum(-1)
    return x
        

# -- LEE FILTER (src : https://github.com/torchgeo/torchgeo/blob/e0b255d52f23e689c5d2a278776f4eb3b5498f32/torchgeo/transforms/sar.py#L97)

def _box_filter(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Apply a per-channel box (mean) filter over the spatial dimensions.

    Args:
        x: Input tensor of shape ``(B, C, H, W)``.
        window_size: Odd integer side length of the smoothing window.

    Returns:
        Smoothed tensor of identical shape.
    """
    pad = window_size // 2
    x_padded = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    kernel = torch.ones(1, 1, window_size, window_size, device=x.device, dtype=x.dtype) / float(window_size * window_size)
    channels = x.shape[1]
    kernel = kernel.expand(channels, 1, window_size, window_size)
    return F.conv2d(x_padded, kernel, groups=channels)

def lee_filter(image: torch.Tensor, window_size: int = 7, num_looks: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    '''Apply a classical Lee filter to a SAR tensor image of shape (B,C,H,W).
    '''
    if window_size < 1 or window_size % 2 == 0: raise ValueError(f'window_size must be a positive odd integer, got {window_size}')
    if num_looks <= 0: raise ValueError(f'num_looks must be > 0, got {num_looks}')

    sigma_v_sq = 1.0 / float(num_looks)

    mean_local = _box_filter(image, window_size)
    mean_sq_local = _box_filter(image * image, window_size)
    var_local = (mean_sq_local - mean_local * mean_local).clamp(min=0.0)

    var_signal = (var_local - sigma_v_sq * mean_local * mean_local).clamp(min=0.0)
    weight = var_signal / (var_signal + sigma_v_sq * mean_local * mean_local + eps)

    return mean_local + weight * (image - mean_local)