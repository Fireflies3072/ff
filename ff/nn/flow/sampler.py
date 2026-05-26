import torch

class FlowEulerSampler:
    def __init__(self, num_steps=20):
        self.num_steps = num_steps

    def step(self, x_t, pred_velocity):
        out = x_t + pred_velocity * (1 / self.num_steps)
        return out

    @torch.no_grad()
    def sample(self, model_fn, x_init, **kwargs):
        b = x_init.shape[0]
        x = x_init
        for i in range(self.num_steps):
            t = i / self.num_steps
            t_batch = torch.full((b,), t, device=x_init.device)
            pred_velocity = model_fn(x, t_batch, **kwargs)
            x = self.step(x, pred_velocity)
        return x
