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

class RotaryPositionEmbedding1d(nn.Module):
    # Attributes
    omega: torch.Tensor

    def __init__(self, head_dim, theta=10000.0):
        """
        1D RoPE: Support floating point coordinate input.
        Args:
            head_dim: Head dimension (must be even)
            theta: Base for frequency calculation
        """
        super().__init__()
        assert head_dim % 2 == 0, "RoPE dim must be even"
        self.head_dim = head_dim
        self.theta = theta
        
        # Calculate frequency terms omega_j (angular frequency)
        # omega_j = theta ** (-2j / dim)
        # Shape: (head_dim // 2,)
        half_dim = head_dim // 2
        omega = theta ** (-torch.arange(0, half_dim).float() / half_dim)
        self.register_buffer("omega", omega, persistent=False)
        
    def _rotate_half(self, x):
        """
        Core operation to simulate complex rotation in real domain.
        [x1, x2] -> [-x2, x1]
        """
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x, t):
        """
        RoPE rotated tensor, same shape as x
        Rotation formula: (x, y) -> (x * cos(θ) - y * sin(θ), y * cos(θ) + x * sin(θ))
        Args:
            x: Input tensor (B, Seq_Len, Num_Heads, Head_Dim)
            t: Coordinate tensor (B, Seq_Len) or (Seq_Len,). Can be floating point (continuous signal).
        Returns:
            Rotated tensor, same shape as x
        """
        # Calculate total angle (angle = t * omega)
        # t: (B, L) or (L,)
        # omega: (D/2,)
        t = t.to(dtype=self.omega.dtype)
        if t.ndim == 1:
            # Broadcast to all Batch
            angle = t[:, None] * self.omega[None, :] # (L, D/2)
            angle = angle[None, :, None, :]          # (1, L, 1, D/2)
        else:
            # For each Batch independently
            angle = t[..., None] * self.omega # (B, L, D/2)
            angle = angle.unsqueeze(2)        # (B, L, 1, D/2)

        # Concatenate angle twice for rotation calculation
        angle = torch.cat((angle, angle), dim=-1) # (1, L, 1, D) or (B, L, 1, D)
        # Calculate cos and sin
        cos = angle.cos()
        sin = angle.sin()
        
        # Apply rotation formula: x*cos + rotate_half(x)*sin
        return (x * cos) + (self._rotate_half(x) * sin)

class RotaryPositionEmbedding2d(nn.Module):
    # Attributes
    omega: torch.Tensor

    def __init__(self, head_dim, theta=10000.0):
        """
        1D RoPE: Support floating point coordinate input.
        Args:
            head_dim: Head dimension (must be divisible by 4)
            theta: Base for frequency calculation
        """
        super().__init__()
        assert head_dim % 4 == 0, "RoPE dim must be divisible by 4"
        self.head_dim = head_dim
        self.theta = theta
        
        # Calculate frequency terms omega_j (angular frequency)
        # omega_j = theta ** (-2j / dim)
        # Shape: (head_dim // 4,)
        quarter_dim = head_dim // 4
        omega = theta ** (-torch.arange(0, quarter_dim).float() / quarter_dim)
        self.register_buffer("omega", omega, persistent=False)
        
    def _rotate_half(self, x):
        """
        Core operation to simulate complex rotation in real domain.
        [x1, x2] -> [-x2, x1]
        """
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x, grid_h, grid_w):
        """
        RoPE rotated tensor, same shape as x
        Rotation formula: (x, y) -> (x * cos(θ) - y * sin(θ), y * cos(θ) + x * sin(θ))
        Args:
            x: Input tensor (B, Seq_Len, Num_Heads, Head_Dim)
            grid_h: Coordinate tensor (B, Seq_Len) or (Seq_Len,). Can be floating point (continuous signal).
            grid_w: Coordinate tensor (B, Seq_Len) or (Seq_Len,). Can be floating point (continuous signal).
        Returns:
            Rotated tensor, same shape as x
        """
        # Calculate total angle (angle = t * omega)
        # grid_h: (B, L) or (L,)
        # grid_w: (B, L) or (L,)
        # omega: (D/4,)
        grid_h = grid_h.to(dtype=self.omega.dtype)
        grid_w = grid_w.to(dtype=self.omega.dtype)
        if grid_h.ndim == 1:
            # Broadcast to all Batch
            angle_h = grid_h[:, None] * self.omega[None, :] # (L, D/4)
            angle_w = grid_w[:, None] * self.omega[None, :] # (L, D/4)
            angle_h = angle_h[None, :, None, :]             # (1, L, 1, D/4)
            angle_w = angle_w[None, :, None, :]             # (1, L, 1, D/4)
        else:
            # For each Batch independently
            angle_h = grid_h[..., None] * self.omega # (B, L, D/4)
            angle_w = grid_w[..., None] * self.omega # (B, L, D/4)
            angle_h = angle_h.unsqueeze(2)           # (B, L, 1, D/4)
            angle_w = angle_w.unsqueeze(2)           # (B, L, 1, D/4)

        # Concatenate angle twice for rotation calculation
        angle_h = torch.cat((angle_h, angle_h), dim=-1) # (1, L, 1, D/2) or (B, L, 1, D/2)
        angle_w = torch.cat((angle_w, angle_w), dim=-1) # (1, L, 1, D/2) or (B, L, 1, D/2)
        # Calculate cos and sin
        cos_h = angle_h.cos()
        sin_h = angle_h.sin()
        cos_w = angle_w.cos()
        sin_w = angle_w.sin()
        
        # Apply rotation formula: x*cos + rotate_half(x)*sin
        x_h = x[..., :x.shape[-1]//2]
        x_w = x[..., x.shape[-1]//2:]
        out_h = (x_h * cos_h) + (self._rotate_half(x_h) * sin_h)
        out_w = (x_w * cos_w) + (self._rotate_half(x_w) * sin_w)
        return torch.cat((out_h, out_w), dim=-1)
