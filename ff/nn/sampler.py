import torch
from scheduler import VPScheduler

class DDPMSampler:
    def __init__(self, scheduler: VPScheduler, limit=None):
        self.scheduler = scheduler
        self.limit = limit

    def step(self, x_t, t, pred_noise):
        b = x_t.shape[0]
        n = len(x_t.shape) - 1
        # Get schedule parameters
        beta = self.scheduler.betas[t-1].reshape(b, *([1] * n))
        alpha_cumprod = self.scheduler.alphas_cumprod[t].reshape(b, *([1] * n))
        alpha_cumprod_prev = self.scheduler.alphas_cumprod[t-1].reshape(b, *([1] * n))
        # Calculate coefficients
        alpha = 1 - beta
        sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod)
        k1 = torch.sqrt(alpha_cumprod_prev) * beta / (1 - alpha_cumprod)
        k2 = torch.sqrt(alpha) * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod)
        posterior_variances = beta * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod)
        sigma = torch.sqrt(posterior_variances)
        # Prepare noise
        mask = (t > 1).reshape(b, *([1] * n))
        noise = torch.randn_like(x_t) * mask

        # Calculate x0_pred
        x0_pred = (x_t - sqrt_one_minus_alpha_cumprod * pred_noise) / sqrt_alpha_cumprod
        if self.limit is not None:
            x0_pred = torch.clamp(x0_pred, self.limit[0], self.limit[1])
        
        # Calculate next step
        out = k1 * x0_pred + k2 * x_t + sigma * noise
        return out

    @torch.no_grad()
    def sample(self, model_fn, x_init, **kwargs):
        b = x_init.shape[0]
        x = x_init
        for t in range(self.scheduler.T, 0, -1):
            t_batch = torch.full((b,), t, device=x_init.device, dtype=torch.long)
            pred_noise = model_fn(x, t_batch-1, **kwargs)
            x = self.step(x, t_batch, pred_noise)
        return x

class DDIMSampler:
    def __init__(self, scheduler: VPScheduler, limit=None, eta=0, num_steps=0):
        self.scheduler = scheduler
        self.limit = limit
        self.eta = eta
        self.num_steps = num_steps if num_steps > 0 else self.scheduler.T

    def step(self, x_t, t, t_prev, pred_noise):
        b = x_t.shape[0]
        n = len(x_t.shape) - 1
        
        # Get schedule parameters
        alpha_cumprod = self.scheduler.alphas_cumprod[t].reshape(b, *([1] * n))
        alpha_cumprod_prev = self.scheduler.alphas_cumprod[t_prev].reshape(b, *([1] * n))
        # Calculate coefficients
        sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod)
        sigma = self.eta * torch.sqrt((1 - alpha_cumprod_prev) / (1 - alpha_cumprod) * (1 - alpha_cumprod / alpha_cumprod_prev))
        k1 = torch.sqrt(alpha_cumprod_prev)
        k2 = torch.sqrt(1 - alpha_cumprod_prev - sigma**2)
        # Prepare noise
        mask = (t_prev > 0).reshape(b, *([1] * n))
        noise = torch.randn_like(x_t) * mask

        # Calculate x0_pred
        x0_pred = (x_t - sqrt_one_minus_alpha_cumprod * pred_noise) / sqrt_alpha_cumprod
        if self.limit is not None:
            x0_pred = torch.clamp(x0_pred, self.limit[0], self.limit[1])
        
        # Calculate next step
        out = k1 * x0_pred + k2 * pred_noise + sigma * noise
        return out

    @torch.no_grad()
    def sample(self, model_fn, x_init, **kwargs):
        b = x_init.shape[0]
        device = x_init.device
        
        times = torch.linspace(0, self.scheduler.T, steps=self.num_steps + 1, device=device).long()
        
        x = x_init
        for i in range(self.num_steps, 0, -1):
            t_batch = torch.full((b,), times[i], device=device, dtype=torch.long)
            t_prev_batch = torch.full((b,), times[i-1], device=device, dtype=torch.long)
            pred_noise = model_fn(x, t_batch-1, **kwargs)
            x = self.step(x, t_batch, t_prev_batch, pred_noise)
            
        return x
