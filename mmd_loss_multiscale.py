import torch

def mmd_loss_multiscale(x_gen, x_real, scales=[0.1, 0.2, 0.5, 1.0, 2.0]):
    """
    Vectorized Multi-Scale MMD Loss.
    """
    # 1. Compute pairwise squared Euclidean distances
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x, y>
    
    xx = torch.mm(x_gen, x_gen.t())
    yy = torch.mm(x_real, x_real.t())
    xy = torch.mm(x_gen, x_real.t())
    
    rx = (x_gen.pow(2).sum(dim=1)).unsqueeze(0)
    ry = (x_real.pow(2).sum(dim=1)).unsqueeze(0)
    
    # Distance matrices (clamped for numerical stability)
    dxx = torch.clamp(rx.t() + rx - 2.0 * xx, min=0.0)
    dyy = torch.clamp(ry.t() + ry - 2.0 * yy, min=0.0)
    dxy = torch.clamp(rx.t() + ry - 2.0 * xy, min=0.0)
    
    loss = 0.0
    for s in scales:
        gamma = 1.0 / (2 * s**2)
        # Kernel values
        k_xx = torch.exp(-gamma * dxx)
        k_yy = torch.exp(-gamma * dyy)
        k_xy = torch.exp(-gamma * dxy)
        
        # Unbiased MMD estimator usually excludes diagonal for k_xx/k_yy
        # but biased is often sufficient for PINN training and more stable.
        loss += k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
        
    return loss