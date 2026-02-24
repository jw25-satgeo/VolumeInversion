'''
COPULA SIMILARITY - GPT-Torched
-- --
CSIM: A Copula-based similarity index sensitive to local changes for Image quality assessment
    Safouane El Ghazouali, Umberto michelucci, Yassin El Hillali, Hichem Nouira (2024)
    [src : https://arxiv.org/abs/2410.01411]
-- --
https://github.com/safouaneelg/copulasimilarity/blob/main/CopulaSimilarity/CSM.py
'''
import torch
import torch.nn.functional as F
import math

def copula_sim(x, patch_size=64, eps=1e-6):
    """
    x: (2, H, W, C)
    returns: (H/P, W/P)
    """
    _, H, W, _ = x.shape
    # (2, H, W, C) → (2, C, H, W)
    x = x.permute(0, 3, 1, 2)
    # (2, C, H, W) → (2, C*P*P, B)
    x = F.unfold(x, kernel_size=patch_size, stride=patch_size)
    # (2, C*P*P, B) → (2, B, P*P, C)
    x = x.transpose(1, 2).view(2, x.shape[-1], patch_size * patch_size, x.shape[1] // (patch_size**2))

    # ranks: (2, B, P*P, C) → (2, B, P*P) - converts mag. distribution to indices of mag. order, e.g. flattened relational
    x = (torch.argsort(torch.argsort(x, dim=2), dim=2) + 1).float().mean(dim=3)

    # normalize to (0,1)
    x = (x / (x.shape[-1] + 1.0)).clamp(eps, 1 - eps)

    # Gaussian copula: (2, B, P*P) - uses inverse G. error function [erfinv] to remap normed mag-relational to gaussian coords
    x = torch.nan_to_num( math.sqrt(2.0)*torch.erfinv(2*x - 1) , nan=0.0, posinf=0.0, neginf=0.0)
    
    # distance (B, P*P) → similarity: (B,)
    x = (1.0 - torch.norm(x[0] - x[1], dim=1) / math.sqrt(H)).clamp(0.0, None)
    
    # (B,) → (H/P, W/P)
    return x.view((H - patch_size) // patch_size + 1,
                  (W - patch_size) // patch_size + 1)