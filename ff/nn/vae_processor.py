import torch
import torch.nn as nn
from diffusers import AutoencoderKL

class VAEProcessor(nn.Module):
    def __init__(self, model_id: str = 'stabilityai/sd-vae-ft-mse'):
        super().__init__()
        self.vae_model = AutoencoderKL.from_pretrained(model_id).eval()
        self.vae_model.requires_grad_(False)
        self.scaling_factor = self.vae_model.config.scaling_factor
    
    @torch.no_grad()
    def encode(self, x: torch.Tensor, use_scaling_factor: bool = True, deterministic: bool = True) -> torch.Tensor:
        latent_dist = self.vae_model.encode(x).latent_dist
        latent = latent_dist.mode() if deterministic else latent_dist.sample()
        if use_scaling_factor:
            latent = latent * self.scaling_factor
        return latent

    @torch.no_grad()
    def decode(self, x: torch.Tensor, use_scaling_factor: bool = True) -> torch.Tensor:
        if use_scaling_factor:
            x = x / self.scaling_factor
        out = self.vae_model.decode(x).sample
        return out
