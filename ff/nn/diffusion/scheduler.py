import torch
from torch import nn

class VPScheduler(nn.Module):
    # Attributes
    betas: torch.Tensor
    alphas_cumprod: torch.Tensor

    def __init__(self, T=1000):
        super().__init__()
        self.T = T

    def add_noise(self, x_0, t):
        b = x_0.shape[0]
        n = len(x_0.shape) - 1
        noise = torch.randn_like(x_0)
        alpha_cumprod = self.alphas_cumprod[t+1].reshape(b, *([1] * n))
        k1 = torch.sqrt(alpha_cumprod)
        k2 = torch.sqrt(1 - alpha_cumprod)
        x_t = k1 * x_0 + k2 * noise
        return x_t, noise
    
    def _calculate_schedule(self, betas):
        # Calculate the schedule parameters
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, 0)
        alphas_cumprod = torch.cat([torch.ones(1, device=betas.device), alphas_cumprod], dim=0)

        # Register the buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)

class LinearScheduler(VPScheduler):
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__(T)
        self.beta_start = beta_start
        self.beta_end = beta_end
        betas = torch.linspace(beta_start, beta_end, self.T)
        self._calculate_schedule(betas)

class CosineScheduler(VPScheduler):
    def __init__(self, T=1000, s=0.008):
        super().__init__(T)
        self.s = s
        x = torch.linspace(0, T, T + 1)
        alphas_cumprod = torch.cos((x / T + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0.0001, 0.999)
        self._calculate_schedule(betas)
