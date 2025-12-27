import torch
import math

class Density3DRegistry:
    """
    Registry for complex 3D density targets.
    Supports GMM (Discontinuous) and Swiss Roll (Manifold).
    """
    def __init__(self, device):
        self.device = device
        
        # GMM Parameters 
        self.gmm_pi = torch.tensor([0.5, 0.3, 0.2], device=device)
        self.gmm_mu = torch.tensor([2.5, 0.0, -1.5], 
            [-2.0, 2.0, 1.0], 
            [0.0, -2.5, 2.0], device=device)
        # Stored as standard deviations for sampling, squared for covariance
        self.gmm_sigma = torch.tensor([0.60, 0.50, 0.70], 
            [0.45, 0.65, 0.40], 
            [0.55, 0.40, 0.60], device=device)

    def sample_gmm(self, batch_size):
        """
        Samples from the 3-component GMM.
        """
        # 1. Choose component
        indices = torch.multinomial(self.gmm_pi, batch_size, replacement=True)
        
        # 2. Gather parameters
        mu = self.gmm_mu[indices]
        sigma = self.gmm_sigma[indices]
        
        # 3. Reparameterization
        epsilon = torch.randn(batch_size, 3, device=self.device)
        return mu + epsilon * sigma

    def sample_swiss_roll(self, batch_size):
        """
        Samples from a normalized 3D Swiss Roll.
        """
        # Standard Swiss Roll logic
        t = 1.5 * math.pi * (1 + 2 * torch.rand(batch_size, device=self.device))
        h = 21 * torch.rand(batch_size, device=self.device)
        
        x = t * torch.cos(t)
        y = h
        z = t * torch.sin(t)
        
        data = torch.stack([x, y, z], dim=1)
        
        # Normalize to be roughly within [-3, 3] to match standard Gaussian scale
        data = (data - data.mean(dim=0)) / data.std(dim=0)
        return data

    def get_score_gmm(self, x):
        """
        Computes the analytical score (grad log p) for the GMM.
        Used for Score-Informed PINGS training.
        """
        # To compute \nabla_x \log \sum \pi_k N(x; \mu_k, \Sigma_k)
        # We use the LogSumExp trick for numerical stability.
        
        # x shape:
        # mu shape: [3, 3]
        # sigma shape: [3, 3]
        
        B, D = x.shape
        K = self.gmm_pi.shape
        
        # Expand x to and mu to for broadcasting
        x_exp = x.unsqueeze(1)
        mu_exp = self.gmm_mu.unsqueeze(0)
        sigma_exp = self.gmm_sigma.unsqueeze(0)
        
        # Compute log-likelihood of each component
        # log N(x; mu, var) = -0.5 * log(2pi) - log(sigma) - 0.5 * ((x-mu)/sigma)^2
        # We ignore constant -0.5 * log(2pi) as it drops out in gradient
        
        diff = x_exp - mu_exp
        exponent = -0.5 * (diff / sigma_exp).pow(2).sum(dim=2) #
        log_coeff = -torch.log(sigma_exp).sum(dim=2) # [1, K] (determinant term)
        log_probs = exponent + log_coeff + torch.log(self.gmm_pi).unsqueeze(0) #
        
        # We need the gradient of: log(sum(exp(log_probs)))
        # PyTorch autograd handles this efficiently if we start with x requiring grad
        
        # NOTE: This function is usually called inside the training loop 
        # where x already requires_grad. If called externally, handle carefully.
        
        # Computation:
        # L = logsumexp(log_probs)
        # grad L = sum ( exp(log_probs_k) / sum(exp(log_probs)) * grad(log_probs_k) )
        # This is effectively computing the responsibility r_k(x) weighted sum.
        
        # However, relying on autograd for the mixture score is safer and cleaner
        # than manually deriving the vector algebra, especially for diagonals.
        return torch.autograd.grad(
            torch.logsumexp(log_probs, dim=1).sum(), 
            x, 
            create_graph=True
        )