import torch
import torch.nn as nn

class ClassConditionedCFGModel(nn.Module):
    def __init__(self, model, num_classes, guidance_scale=7.0):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.guidance_scale = guidance_scale

    def forward(self, x, t, label):
        bs = x.shape[0]
        label_u = torch.full((bs,), self.num_classes, dtype=torch.long, device=x.device)
        # Prepare double input for CFG
        x_double = torch.cat([x, x], dim=0)
        t_double = torch.cat([t, t], dim=0)
        label_double = torch.cat([label, label_u], dim=0)

        # Model prediction
        out_double = self.model(x_double, t_double, label_double)
        out_c, out_u = torch.chunk(out_double, 2, dim=0)

        # CFG noise prediction
        return out_u + self.guidance_scale * (out_c - out_u)
