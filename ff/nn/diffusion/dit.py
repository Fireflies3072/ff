import torch
import torch.nn as nn
import torch.nn.functional as F

from ..architecture import *

class DiTGeneric(nn.Module):
    def __init__(self, size: int, patch_size: int, num_channels: int, dim: int,
                 num_layers: int, num_heads: int, use_resolution: bool, ndim: int):
        super().__init__()
        self.size = size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.dim = dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.use_resolution = use_resolution
        self.ndim = ndim
        
        # Patch
        patch_class = getattr(nn, f"DiTPatchBlock{ndim}d")
        self.patch = patch_class(patch_size, num_channels, dim)
        unpatch_class = getattr(nn, f"DiTUnpatchBlock{ndim}d")
        self.unpatch = unpatch_class(patch_size, num_channels, dim)
        
        # Time Embedding
        self.time_embedding = nn.Sequential(
            SinusoidalEmbedding(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        # Resolution Embedding
        if use_resolution:
            resolution_embedding_class = getattr(nn, f"ResolutionEmbedding{ndim}d")
            self.resolution_embedding = resolution_embedding_class(dim)
        
        # RoPE
        rope_class = getattr(nn, f"RotaryPositionEmbedding{ndim}d")
        self.rope = rope_class(dim // num_heads)
        
        # DiT Blocks
        self.blocks = None

    def forward(self, x, t, **kwargs):
        # Size
        b, c, *spatial_shape = x.shape
        grid_axes = [torch.arange(s, dtype=x.dtype, device=x.device) for s in spatial_shape]
        grid = torch.meshgrid(*grid_axes, indexing='ij')
        grid = [g.reshape(-1) for g in grid]

        # Condition
        embedding = self._get_embedding(x, t, **kwargs)
        # Patch
        x = self.patch(x)
        # Blocks
        for block in self.blocks:
            x = block(x, embedding, self.rope, grid)
        # Unpatch
        out = self.unpatch(x, spatial_shape)
        
        return out
    
    def _get_embedding(self, x, t, **kwargs):
        t_embedding = self.time_embedding(t)
        if not self.use_resolution:
            embedding = t_embedding
        else:
            r_embedding = self.resolution_embedding(x)
            embedding = t_embedding + r_embedding
        return embedding

class UnconditionalDiTGeneric(DiTGeneric):
    def __init__(self, size: int, patch_size: int, num_channels: int, dim: int,
                 num_layers: int, num_heads: int, use_resolution: bool, ndim: int):
        super().__init__(size, patch_size, num_channels, dim,
                         num_layers, num_heads, use_resolution, ndim)

        # DiT Blocks
        self.blocks = nn.ModuleList([DiTBlock(dim, num_heads) for _ in range(num_layers)])
    
    def forward(self, x, t):
        return super().forward(x, t)

class UnconditionalDiT1d(UnconditionalDiTGeneric):
    def __init__(self, size=32, patch_size=2, num_channels=3, dim=512,
                 num_layers=12, num_heads=8, use_resolution=False):
        super().__init__(size, patch_size, num_channels, dim,
                         num_layers, num_heads, use_resolution=False, ndim=1)

class UnconditionalDiT2d(UnconditionalDiTGeneric):
    def __init__(self, size=32, patch_size=2, num_channels=3, dim=512,
                 num_layers=12, num_heads=8, use_resolution=False):
        super().__init__(size, patch_size, num_channels, dim,
                         num_layers, num_heads, use_resolution=False, ndim=2)

class UnconditionalDiT3d(UnconditionalDiTGeneric):
    def __init__(self, size=32, patch_size=2, num_channels=3, dim=512,
                 num_layers=12, num_heads=8, use_resolution=False):
        super().__init__(size, patch_size, num_channels, dim,
                         num_layers, num_heads, use_resolution=False, ndim=3)

class ClassConditionalDiTGeneric(DiTGeneric):
    def __init__(self, size: int, patch_size: int, num_channels: int, dim: int,
                 num_layers: int, num_heads: int, num_classes: int, use_resolution: bool, ndim: int):
        super().__init__(size, patch_size, num_channels, dim,
                         num_layers, num_heads, use_resolution, ndim)
        self.num_classes = num_classes

        # DiT Blocks
        self.blocks = nn.ModuleList([DiTBlock(dim, num_heads) for _ in range(num_layers)])

        # Label Embedding
        self.label_embedding = nn.Embedding(num_classes, dim)
    
    def forward(self, x, t, label):
        return super().forward(x, t, label=label)
    
    def _get_embedding(self, x, t, label):
        l_embedding = self.label_embedding(label)
        embedding = super()._get_embedding(x, t) + l_embedding
        return embedding

class ClassConditionalDiT1d(ClassConditionalDiTGeneric):
    def __init__(self, size=32, patch_size=2, num_channels=3, dim=512,
                 num_layers=12, num_heads=8, num_classes=10, use_resolution=False):
        super().__init__(size, patch_size, num_channels, dim,
                         num_layers, num_heads, num_classes, use_resolution, ndim=1)

class ClassConditionalDiT2d(ClassConditionalDiTGeneric):
    def __init__(self, size=32, patch_size=2, num_channels=3, dim=512,
                 num_layers=12, num_heads=8, num_classes=10, use_resolution=False):
        super().__init__(size, patch_size, num_channels, dim,
                         num_layers, num_heads, num_classes, use_resolution, ndim=2)

class ClassConditionalDiT3d(ClassConditionalDiTGeneric):
    def __init__(self, size=32, patch_size=2, num_channels=3, dim=512,
                 num_layers=12, num_heads=8, num_classes=10, use_resolution=False):
        super().__init__(size, patch_size, num_channels, dim,
                         num_layers, num_heads, num_classes, use_resolution, ndim=3)

class ContextConditionalDiTGeneric(DiTGeneric):
    def __init__(self, size: int, patch_size: int, num_channels: int, dim: int,
                 num_layers: int, num_heads: int, use_resolution: bool, ndim: int):
        super().__init__(size, patch_size, num_channels, dim,
                         num_layers, num_heads, use_resolution, ndim)

        # DiT Blocks
        self.blocks = nn.ModuleList([DiTBlockWithCrossAttention(dim, num_heads) for _ in range(num_layers)])
    
    def forward(self, x, t, context, mask=None):
        # Size
        b, c, *spatial_shape = x.shape
        grid_axes = [torch.arange(s, dtype=x.dtype, device=x.device) for s in spatial_shape]
        grid = torch.meshgrid(*grid_axes, indexing='ij')
        grid = [g.reshape(-1) for g in grid]

        # Condition
        embedding = self._get_embedding(x, t)
        # Patch
        x = self.patch(x)
        # Blocks
        for block in self.blocks:
            x = block(x, embedding, context, self.rope, grid, mask)
        # Unpatch
        out = self.unpatch(x, spatial_shape)
        
        return out

class ContextConditionalDiT1d(ContextConditionalDiTGeneric):
    def __init__(self, size=32, patch_size=2, num_channels=3, dim=512,
                 num_layers=12, num_heads=8, use_resolution=False):
        super().__init__(size, patch_size, num_channels, dim,
                         num_layers, num_heads, use_resolution=False, ndim=1)

class ContextConditionalDiT2d(ContextConditionalDiTGeneric):
    def __init__(self, size=32, patch_size=2, num_channels=3, dim=512,
                 num_layers=12, num_heads=8, use_resolution=False):
        super().__init__(size, patch_size, num_channels, dim,
                         num_layers, num_heads, use_resolution=False, ndim=2)

class ContextConditionalDiT3d(ContextConditionalDiTGeneric):
    def __init__(self, size=32, patch_size=2, num_channels=3, dim=512,
                 num_layers=12, num_heads=8, use_resolution=False):
        super().__init__(size, patch_size, num_channels, dim,
                         num_layers, num_heads, use_resolution=False, ndim=3)
