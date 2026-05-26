import torch
import torch.nn as nn

class SinusoidalEmbedding(nn.Module):
    # Attributes
    omega: torch.Tensor

    def __init__(self, dim, theta=10000.0):
        super().__init__()
        self.dim = dim

        half_dim = dim // 2
        omega = theta ** (-torch.arange(0, half_dim).float() / half_dim)
        self.register_buffer("omega", omega, persistent=False)

    def forward(self, t):
        """
        Args:
            t: Time tensor (B,)
        Returns:
            Embedding tensor (B, Dim)
        """
        t = t.to(dtype=self.omega.dtype)
        angle = t[:, None] * self.omega[None, :] # (B, D/2)
        embedding = torch.cat((angle.cos(), angle.sin()), dim=-1) # (B, D)
        return embedding

class RotaryPositionEmbeddingGeneric(nn.Module):
    # Attributes
    omega: torch.Tensor

    def __init__(self, head_dim: int, theta: float, ndim: int):
        """
        Generic RoPE: Support floating point coordinate input.
        Args:
            head_dim: Head dimension (must be divisible by 2 * ndim)
            theta: Base for frequency calculation
            ndim: Number of dimensions
        """
        super().__init__()
        assert head_dim % (2 * ndim) == 0, f"head_dim ({head_dim}) must be divisible by {2 * ndim}"
        self.head_dim = head_dim
        self.theta = theta
        self.ndim = ndim

        self.axis_dim = head_dim // ndim
        
        # Calculate frequency terms omega_i (angular frequency)
        # omega_i = theta ** (-2i / half_axis_dim)
        # Shape: (axis_dim/2,)
        half_axis_dim = self.axis_dim // 2
        omega = theta ** (-torch.arange(0, half_axis_dim).float() / half_axis_dim)
        self.register_buffer("omega", omega, persistent=False)
        
    def _rotate_half(self, x):
        """
        Core operation to simulate complex rotation in real domain.
        [x1, x2] -> [-x2, x1]
        """
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x, grid):
        """
        RoPE rotated tensor, same shape as x
        Rotation formula: (x, y) -> (x * cos(θ) - y * sin(θ), y * cos(θ) + x * sin(θ))
        Args:
            x: Input tensor (B, Num_Heads, Seq_Len, Head_Dim)
            grid: List of ndim coordinate tensors, each of shape (B, Seq_Len) or (Seq_Len,).
                  Can be floating point (continuous signal) or integer (discrete signal).
        Returns:
            Rotated tensor, same shape as x
        """
        # Calculate total angle (angle = t * omega)
        # grid: [(B, L) or (L,), ...]
        # omega: (axis_dim / 2,)
        angles = []
        for g in grid:
            g = g.to(dtype=self.omega.dtype)
            if g.ndim == 1:
                # Broadcast to all Batch
                angle = g[:, None] * self.omega[None, :] # (L, axis_dim/2)
                angle = angle[None, None, :, :]          # (1, 1, L, axis_dim/2)
            else:
                # For each Batch independently
                angle = g[..., None] * self.omega # (B, L, axis_dim/2)
                angle = angle.unsqueeze(1)        # (B, 1, L, axis_dim/2)
            # Concatenate angle twice for rotation calculation
            angle = torch.cat((angle, angle), dim=-1) # (1, 1, L, axis_dim) or (B, 1, L, axis_dim)
            angles.append(angle)
        
        # Split x into ndim parts
        x_parts = x.chunk(self.ndim, dim=-1) # List of (B, H, L, axis_dim)
        
        # Apply rotation formula: x*cos + rotate_half(x)*sin
        outs = []
        for x_part, angle in zip(x_parts, angles):
            cos = angle.cos() # (1, 1, L, axis_dim) or (B, 1, L, axis_dim)
            sin = angle.sin() # (1, 1, L, axis_dim) or (B, 1, L, axis_dim)
            out_part = (x_part * cos) + (self._rotate_half(x_part) * sin) # (B, H, L, axis_dim)
            outs.append(out_part)
        return torch.cat(outs, dim=-1)

class RotaryPositionEmbedding1d(RotaryPositionEmbeddingGeneric):
    def __init__(self, head_dim, theta=10000.0):
        super().__init__(head_dim, theta=theta, ndim=1)

class RotaryPositionEmbedding2d(RotaryPositionEmbeddingGeneric):
    def __init__(self, head_dim, theta=10000.0):
        super().__init__(head_dim, theta=theta, ndim=2)

class RotaryPositionEmbedding3d(RotaryPositionEmbeddingGeneric):
    def __init__(self, head_dim, theta=10000.0):
        super().__init__(head_dim, theta=theta, ndim=3)
