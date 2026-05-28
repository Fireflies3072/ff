import torch
import torch.nn as nn
from beartype import beartype

class CFGScaleGeneric:
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def post_process(self, out_cfg: torch.Tensor, out_c: torch.Tensor) -> torch.Tensor:
        return out_cfg

@beartype
class CFGModel(nn.Module):
    def __init__(self, model, guidance_scale: float | CFGScaleGeneric = 7.0):
        """
        Classifier-Free Guidance (CFG) wrapper for a diffusion model.

        This wrapper handles the dual forward pass (unconditional and conditional)
        and combines them using the guidance scale.

        Args:
            model: The base diffusion model.
            guidance_scale: The scale for guidance. Can be a fixed float or a dynamic
                scale scheduler (CFGScaleGeneric).

        Forward Args:
            x: Input tensor.
            t: Time tensor.
            **kwargs: Additional parameters for the model. If a parameter is a tuple,
                it is interpreted as (unconditional, conditional). Other parameters
                are treated as regular parameters (duplicated for both passes).
        """
        super().__init__()
        self.model = model
        if isinstance(guidance_scale, (int, float)):
            guidance_scale = CFGConstantScale(float(guidance_scale))
        self.guidance_scale = guidance_scale

    def forward(self, x, t, **kwargs):
        # Get guidance scale
        guidance_scale = self.guidance_scale(t).view(-1, *([1] * (x.ndim - 1)))
        
        # Prepare double input for CFG
        x_double = torch.cat([x, x], dim=0)
        t_double = torch.cat([t, t], dim=0)
        kwargs_double = {}
        for key, value in kwargs.items():
            if isinstance(value, tuple):
                kwargs_double[key] = torch.cat([value[0], value[1]], dim=0)
            else:
                kwargs_double[key] = torch.cat([value, value], dim=0)
        # Model prediction
        out_double = self.model(x_double, t_double, **kwargs_double)
        out_u, out_c = torch.chunk(out_double, 2, dim=0)

        # CFG noise prediction
        out_cfg = out_u + guidance_scale * (out_c - out_u)
        return self.post_process(out_cfg, out_c)

    def post_process(self, out_cfg: torch.Tensor, out_c: torch.Tensor) -> torch.Tensor:
        """
        Apply post-processing to the CFG output.
        
        If `guidance_scale` is a scheduler, it delegates to its `post_process` method.
        """
        if isinstance(self.guidance_scale, CFGScaleGeneric):
            return self.guidance_scale.post_process(out_cfg, out_c)
        return out_cfg

@beartype
class CFGConstantScale(CFGScaleGeneric):
    def __init__(self, scale: float=7.0):
        self.scale = scale
    
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return torch.full_like(t, self.scale)

@beartype
class CFGLinearScale(CFGScaleGeneric):
    def __init__(self, scale_max: float=7.0, scale_min: float=1.0):
        """
        Linear CFG scale scheduler.

        The scale decreases linearly from `scale_max` at t=0 to `scale_min` at t=1.

        Args:
            scale_max: Maximum scale at t=0.
            scale_min: Minimum scale at t=1.
        """
        self.scale_max = scale_max
        self.scale_min = scale_min

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        # t=0 -> s_max; t=1 -> s_min
        return self.scale_max + (self.scale_min - self.scale_max) * t

@beartype
class CFGCosineScale(CFGScaleGeneric):
    def __init__(self, scale_max: float=7.0, scale_min: float=1.0):
        """
        Cosine CFG scale scheduler.

        The scale follows a cosine schedule between `scale_max` (at t=0) and `scale_min` (at t=1).

        Args:
            scale_max: Maximum scale at t=0.
            scale_min: Minimum scale at t=1.
        """
        self.scale_max = scale_max
        self.scale_min = scale_min

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        alpha = torch.cos(t * (torch.pi / 2))
        return self.scale_min + (self.scale_max - self.scale_min) * alpha

@beartype
class CFGIntervalScale(CFGScaleGeneric):
    def __init__(self, scale_max: float=7.0, scale_min: float=1.0, t_threshold: float = 0.9):
        """
        Interval-based CFG scale scheduler.

        Uses `scale_max` when t <= `t_threshold`, and `scale_min` otherwise.

        Args:
            scale_max: Scale used before the threshold.
            scale_min: Scale used after the threshold.
            t_threshold: The time threshold for switching scales.
        """
        self.scale_max = scale_max
        self.scale_min = scale_min
        self.t_threshold = t_threshold

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        scale = torch.ones_like(t) * self.scale_max
        scale[t > self.t_threshold] = self.scale_min
        return scale

@beartype
class CFGCorrectedScale(CFGScaleGeneric):
    def __init__(self, scale: float=7.0, factor: float=0.7):
        """
        Corrected CFG scale with standard deviation rescaling.

        Maintains a constant scale but applies a post-processing step to rescale
        the CFG output's standard deviation to match the conditional output's
        standard deviation, helping to prevent over-saturation.

        Args:
            scale: The fixed guidance scale.
            factor: Mixing factor between rescaled and original CFG output.
        """
        self.scale = scale
        self.factor = factor

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return torch.full_like(t, self.scale)

    def post_process(self, out_cfg: torch.Tensor, out_c: torch.Tensor) -> torch.Tensor:
        # Calculate standard deviation
        std_c = out_c.std(dim=list(range(1, out_c.ndim)), keepdim=True)
        std_cfg = out_cfg.std(dim=list(range(1, out_cfg.ndim)), keepdim=True)
        
        # Calculate corrected result
        # Formula: out_cfg * (std_c / std_cfg)
        rescaled_out = out_cfg * (std_c / (std_cfg + 1e-6))
        
        # Mix original CFG and corrected result
        return self.factor * rescaled_out + (1.0 - self.factor) * out_cfg
