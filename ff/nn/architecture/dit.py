import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from .time_embedding import SinusoidalEmbedding

class DiTPatchBlockGeneric(nn.Module):
    def __init__(self, patch_size: Union[int, tuple[int, ...]], num_channels: int, dim: int, ndim: int):
        super().__init__()
        if isinstance(patch_size, (list, tuple)):
            assert len(patch_size) == ndim, f"patch_size length must match ndim {ndim}"
            self.patch_size = tuple(patch_size)
        else:
            self.patch_size = (patch_size,) * ndim
        self.num_channels = num_channels
        self.dim = dim
        self.ndim = ndim
        
        # Dynamic convolution class (nn.Conv1d, nn.Conv2d, nn.Conv3d)
        conv_class = getattr(nn, f"Conv{ndim}d")
        self.conv = conv_class(num_channels, dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        x = self.conv(x) # (b, d, w/p) or (b, d, h/p, w/p) or (b, d, depth/p, h/p, w/p)
        x = x.flatten(2) # (b, d, l)
        x = x.transpose(1, 2) # (b, l, d)
        return x

class DiTPatchBlock1d(DiTPatchBlockGeneric):
    def __init__(self, patch_size: Union[int, tuple[int, ...]], num_channels: int, dim: int):
        super().__init__(patch_size, num_channels, dim, ndim=1)

class DiTPatchBlock2d(DiTPatchBlockGeneric):
    def __init__(self, patch_size: Union[int, tuple[int, ...]], num_channels: int, dim: int):
        super().__init__(patch_size, num_channels, dim, ndim=2)

class DiTPatchBlock3d(DiTPatchBlockGeneric):
    def __init__(self, patch_size: Union[int, tuple[int, ...]], num_channels: int, dim: int):
        super().__init__(patch_size, num_channels, dim, ndim=3)

class DiTUnpatchBlockGeneric(nn.Module):
    def __init__(self, patch_size: Union[int, tuple[int, ...]], num_channels: int, dim: int, ndim: int):
        super().__init__()
        if isinstance(patch_size, (list, tuple)):
            assert len(patch_size) == ndim, f"patch_size length must match ndim {ndim}"
            self.patch_size = tuple(patch_size)
        else:
            self.patch_size = (patch_size,) * ndim
        self.num_channels = num_channels
        self.dim = dim
        self.ndim = ndim

        self.projection = nn.Sequential(
            nn.LayerNorm(dim, elementwise_affine=False),
            nn.Linear(dim, num_channels * (patch_size**ndim))
        )

    def forward(self, x: torch.Tensor, spatial_shape: tuple, grid_shape: tuple):
        # x: (b, l, d)
        b = x.shape[0]
        c = self.num_channels
        
        x = self.projection(x) # (b, l, c * p^ndim)

        # To (b, w/p, p, c) or (b, h/p, w/p, p, p, c) or (b, depth/p, h/p, w/p, p, p, p, c)
        x = x.reshape(b, *grid_shape, *self.patch_size, c)
        
        # To (b, c, w/p, p) or (b, c, h/p, p, w/p, p) or (b, c, depth/p, p, h/p, p, w/p, p)
        permute_order = [0, 2 * self.ndim + 1]
        for i in range(1, self.ndim + 1):
            permute_order.append(i)
            permute_order.append(self.ndim + i)
        x = x.permute(*permute_order).contiguous()
        
        out = x.reshape(b, c, *spatial_shape) # (b, c, w) or (b, c, h, w) or (b, c, depth, h, w)
        return out

class DiTUnpatchBlock1d(DiTUnpatchBlockGeneric):
    def __init__(self, patch_size: Union[int, tuple[int, ...]], num_channels: int, dim: int):
        super().__init__(patch_size, num_channels, dim, ndim=1)

class DiTUnpatchBlock2d(DiTUnpatchBlockGeneric):
    def __init__(self, patch_size: Union[int, tuple[int, ...]], num_channels: int, dim: int):
        super().__init__(patch_size, num_channels, dim, ndim=2)

class DiTUnpatchBlock3d(DiTUnpatchBlockGeneric):
    def __init__(self, patch_size: Union[int, tuple[int, ...]], num_channels: int, dim: int):
        super().__init__(patch_size, num_channels, dim, ndim=3)

class ResolutionEmbeddingGeneric(nn.Module):
    def __init__(self, dim: int, theta: float, ndim: int):
        """
        Args:
            dim: Output dimension
            theta: Sinusoidal embedding theta
            ndim: Number of dimensions
        """
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.ndim = ndim
        
        self.sin_embedding = SinusoidalEmbedding(dim, theta)
        self.projection = nn.Sequential(
            nn.Linear(dim * ndim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input tensor (b, c, w) or (b, c, h, w) or (b, c, depth, h, w)
        """
        # Convert Python integers to Tensor
        b, c, *spatial_shape = x.shape
        spatial_tensor = [torch.tensor([float(s)], dtype=x.dtype, device=x.device) for s in spatial_shape]
        spatial_embedding = [self.sin_embedding(s) for s in spatial_tensor]
        resolution_embedding = torch.cat(spatial_embedding, dim=-1) # (1, dim * ndim)
        resolution_embedding = self.projection(resolution_embedding) # (1, dim)
        out = resolution_embedding.repeat(b, 1) # (b, dim)
        return out

class ResolutionEmbedding1d(ResolutionEmbeddingGeneric):
    def __init__(self, dim: int, theta: float=10000.0):
        super().__init__(dim, theta, ndim=1)

class ResolutionEmbedding2d(ResolutionEmbeddingGeneric):
    def __init__(self, dim: int, theta: float=10000.0):
        super().__init__(dim, theta, ndim=2)

class ResolutionEmbedding3d(ResolutionEmbeddingGeneric):
    def __init__(self, dim: int, theta: float=10000.0):
        super().__init__(dim, theta, ndim=3)

class DiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.norm_msa = nn.LayerNorm(dim, elementwise_affine=False)
        self.qkv_msa = nn.Linear(dim, dim * 3)
        self.proj_msa = nn.Linear(dim, dim)
        
        self.norm_mlp = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        
        # AdaLN modulation: predict shift, scale, gate
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim)
        )
        # Initialize the last Linear layer to 0, implement AdaLN-Zero
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, embedding, rope, grid):
        # Modulation parameters
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(embedding).chunk(6, dim=1) # (b, 6 * d)

        # Self-attention and MLP
        x = self._self_attention(x, shift_msa, scale_msa, gate_msa, rope, grid)
        x = self._mlp(x, shift_mlp, scale_mlp, gate_mlp)
        return x
    
    def _self_attention(self, x, shift_msa, scale_msa, gate_msa, rope, grid):
        b, l, d = x.shape
        # Modulate x for multi-head self-attention
        x_msa = self.norm_msa(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1) # (b, l, d)
        # Generate QKV
        qkv = self.qkv_msa(x_msa).reshape(b, l, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # (b, h, l, d)
        # Apply RoPE
        q = rope(q, grid)
        k = rope(k, grid)
        # Multi-head self-attention
        attention = F.scaled_dot_product_attention(q, k, v) # (b, h, l, d)
        attention = attention.transpose(1, 2).reshape(b, l, d)
        x = x + gate_msa.unsqueeze(1) * self.proj_msa(attention)
        return x
    
    def _mlp(self, x, shift_mlp, scale_mlp, gate_mlp):
        # Modulate x for MLP
        x_mlp = self.norm_mlp(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1) # (b, l, d)
        # MLP
        x_mlp = self.proj_mlp(x_mlp)
        x = x + gate_mlp.unsqueeze(1) * x_mlp
        return x

class DiTBlockWithCrossAttention(DiTBlock):
    def __init__(self, dim: int, num_heads: int):
        super().__init__(dim, num_heads)
        self.norm_mca = nn.LayerNorm(dim, elementwise_affine=False)
        self.q_mca = nn.Linear(dim, dim)
        self.kv_mca = nn.Linear(dim, dim * 2)
        self.proj_mca = nn.Linear(dim, dim)
        
        # AdaLN modulation: predict shift, scale, gate
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 9 * dim)
        )
        # Initialize the last Linear layer to 0, implement AdaLN-Zero
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, embedding, context, rope, grid, mask=None):
        # Modulation parameters
        shift_msa, scale_msa, gate_msa, shift_mca, scale_mca, gate_mca, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(embedding).chunk(9, dim=1) # (b, 9 * d)

        # Self-attention and MLP
        x = self._self_attention(x, shift_msa, scale_msa, gate_msa, rope, grid)
        x = self._cross_attention(x, shift_mca, scale_mca, gate_mca, context, mask)
        x = self._mlp(x, shift_mlp, scale_mlp, gate_mlp)
        return x
    
    def _cross_attention(self, x, shift_mca, scale_mca, gate_mca, context, mask):
        b, l, d = x.shape
        m = context.shape[1]
        # Modulate x for multi-head cross-attention
        x_mca = self.norm_mca(x) * (1 + scale_mca.unsqueeze(1)) + shift_mca.unsqueeze(1) # (b, l, d)
        # Generate QKV
        q = self.q_mca(x_mca).reshape(b, l, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv_mca(context).reshape(b, m, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1] # (b, h, m, d)
        # Prepare mask
        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask.to(torch.bool)
            mask = mask.view(b, 1, 1, m)
        # Multi-head cross-attention
        attention = F.scaled_dot_product_attention(q, k, v, attn_mask=mask) # (b, h, l, d)
        attention = attention.transpose(1, 2).reshape(b, l, d)
        x = x + gate_mca.unsqueeze(1) * self.proj_mca(attention)
        return x

class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        """
        Time embedding for DiT.
        Args:
            dim: Output dimension
        """
        super().__init__()
        self.dim = dim
        self.time_embedding = nn.Sequential(
            SinusoidalEmbedding(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
    
    def forward(self, t: torch.Tensor):
        return self.time_embedding(t)
