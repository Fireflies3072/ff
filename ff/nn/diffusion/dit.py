import torch
import torch.nn as nn
from typing import Union

from .. import architecture

class DiTBaseGeneric(nn.Module):
    def __init__(self, patch_size: Union[int, tuple[int, ...]], num_channels: int, dim: int,
                 num_layers: int, num_heads: int, conditioners: dict[str, nn.Module], ndim: int):
        super().__init__()
        if isinstance(patch_size, (list, tuple)):
            assert len(patch_size) == ndim, f"patch_size length must match ndim {ndim}"
            self.patch_size = tuple(patch_size)
        else:
            self.patch_size = (patch_size,) * ndim
        self.num_channels = num_channels
        self.dim = dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.conditioners = nn.ModuleDict(conditioners)
        self.ndim = ndim
        
        # Patch
        patch_class = getattr(architecture, f"DiTPatchBlock{ndim}d")
        self.patch = patch_class(patch_size, num_channels, dim)
        unpatch_class = getattr(architecture, f"DiTUnpatchBlock{ndim}d")
        self.unpatch = unpatch_class(patch_size, num_channels, dim)
        
        # Conditioners
        if self.conditioners is None:
            self.conditioners = {}
        # Time Embedding
        if 'time' in self.conditioners and isinstance(self.conditioners['time'], nn.Module):
            pass
        else:
            self.conditioners['time'] = architecture.TimeEmbedding(dim)
        # Resolution Embedding
        if 'resolution' in self.conditioners and not isinstance(self.conditioners['resolution'], nn.Module):
            resolution_embedding_class = getattr(architecture, f"ResolutionEmbedding{ndim}d")
            self.conditioners['resolution'] = resolution_embedding_class(dim)
        
        # RoPE
        rope_class = getattr(architecture, f"RotaryPositionEmbedding{ndim}d")
        self.rope = rope_class(dim // num_heads)
        
        # DiT Blocks
        self.blocks = None

    def forward(self, x, t, **kwargs):
        # Size
        b, c, *spatial_shape = x.shape
        grid_shape = tuple(s // p for s, p in zip(spatial_shape, self.patch_size))
        grid_axes = [torch.arange(s, dtype=x.dtype, device=x.device) for s in grid_shape]
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
        out = self.unpatch(x, spatial_shape, grid_shape)
        
        return out
    
    def _get_embedding(self, x, t, **kwargs):
        embedding = self.conditioners['time'](t)
        for key, conditioner in self.conditioners.items():
            if key == 'time':
                continue
            elif key == 'resolution':
                embedding += conditioner(x)
            else:
                embedding += conditioner(kwargs[key])
        return embedding

class DiTGeneric(DiTBaseGeneric):
    def __init__(self, patch_size: Union[int, tuple[int, ...]], num_channels: int, dim: int,
                 num_layers: int, num_heads: int, conditioners: dict[str, nn.Module], ndim: int):
        super().__init__(patch_size, num_channels, dim, num_layers, num_heads, conditioners, ndim)
        
        # DiT Blocks
        self.blocks = nn.ModuleList([architecture.DiTBlock(dim, num_heads) for _ in range(num_layers)])

class DiT1d(DiTGeneric):
    def __init__(self, patch_size: Union[int, tuple[int, ...]], num_channels: int, dim: int=768,
                 num_layers: int=12, num_heads: int=12, conditioners: dict[str, nn.Module]=None):
        super().__init__(patch_size, num_channels, dim, num_layers, num_heads, conditioners, ndim=1)

class DiT2d(DiTGeneric):
    def __init__(self, patch_size: Union[int, tuple[int, ...]], num_channels: int, dim: int=768,
                 num_layers: int=12, num_heads: int=12, conditioners: dict[str, nn.Module]=None):
        super().__init__(patch_size, num_channels, dim, num_layers, num_heads, conditioners, ndim=2)

class DiT3d(DiTGeneric):
    def __init__(self, patch_size: Union[int, tuple[int, ...]], num_channels: int, dim: int=768,
                 num_layers: int=12, num_heads: int=12, conditioners: dict[str, nn.Module]=None):
        super().__init__(patch_size, num_channels, dim, num_layers, num_heads, conditioners, ndim=3)

class ContextConditionalDiTGeneric(DiTBaseGeneric):
    def __init__(self, patch_size: Union[int, tuple[int, ...]], num_channels: int, dim: int,
                 num_layers: int, num_heads: int, conditioners: dict[str, nn.Module], ndim: int):
        super().__init__(patch_size, num_channels, dim, num_layers, num_heads, conditioners, ndim)

        # DiT Blocks
        self.blocks = nn.ModuleList(
            [architecture.DiTBlockWithCrossAttention(dim, num_heads) for _ in range(num_layers)]
        )
    
    def forward(self, x, t, context, mask=None, **kwargs):
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
            x = block(x, embedding, context, self.rope, grid, mask)
        # Unpatch
        out = self.unpatch(x, spatial_shape)
        
        return out

class ContextConditionalDiT1d(ContextConditionalDiTGeneric):
    def __init__(self, patch_size: Union[int, tuple[int, ...]], num_channels: int, dim: int=768,
                 num_layers: int=12, num_heads: int=12, conditioners: dict[str, nn.Module]=None):
        super().__init__(patch_size, num_channels, dim, num_layers, num_heads, conditioners, ndim=1)

class ContextConditionalDiT2d(ContextConditionalDiTGeneric):
    def __init__(self, patch_size: Union[int, tuple[int, ...]], num_channels: int, dim: int=768,
                 num_layers: int=12, num_heads: int=12, conditioners: dict[str, nn.Module]=None):
        super().__init__(patch_size, num_channels, dim, num_layers, num_heads, conditioners, ndim=2)

class ContextConditionalDiT3d(ContextConditionalDiTGeneric):
    def __init__(self, patch_size: Union[int, tuple[int, ...]], num_channels: int, dim: int=768,
                 num_layers: int=12, num_heads: int=12, conditioners: dict[str, nn.Module]=None):
        super().__init__(patch_size, num_channels, dim, num_layers, num_heads, conditioners, ndim=3)
