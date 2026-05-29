import torch
import torch.nn as nn
from beartype import beartype

from .. import architecture

class DiTBaseGeneric(nn.Module):
    @beartype
    def __init__(self, patch_size: int|tuple[int, ...], num_channels: int, dim: int,
                 num_layers: int, num_heads: int, conditioners: dict[str, nn.Module]|None, ndim: int):
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
        self.conditioners = conditioners
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
        # Convert to ModuleDict
        self.conditioners = nn.ModuleDict(self.conditioners)
        
        # RoPE
        rope_class = getattr(architecture, f"RotaryPositionEmbedding{ndim}d")
        self.rope = rope_class(dim // num_heads)
        
        # DiT Blocks
        self.blocks = None

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs):
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
        x = self._forward_blocks(x, embedding, grid, **kwargs)
        # Unpatch
        out = self.unpatch(x, spatial_shape, grid_shape)
        
        return out
    
    def _get_embedding(self, x: torch.Tensor, t: torch.Tensor, **kwargs):
        embedding = self.conditioners['time'](t)
        for key, conditioner in self.conditioners.items():
            if key == 'time':
                continue
            elif key == 'resolution':
                embedding = embedding + conditioner(x)
            else:
                embedding = embedding + conditioner(kwargs[key])
        return embedding
    
    def _forward_blocks(self, x: torch.Tensor, embedding: torch.Tensor, grid: list[torch.Tensor], **kwargs):
        raise NotImplementedError('Subclasses must implement this method')

class DiTGeneric(DiTBaseGeneric):
    @beartype
    def __init__(self, patch_size: int|tuple[int, ...], num_channels: int, dim: int,
                 num_layers: int, num_heads: int, conditioners: dict[str, nn.Module]|None, ndim: int):
        super().__init__(patch_size, num_channels, dim, num_layers, num_heads, conditioners, ndim)
        
        # DiT Blocks
        self.blocks = nn.ModuleList([architecture.DiTBlock(dim, num_heads) for _ in range(num_layers)])
    
    def _forward_blocks(self, x: torch.Tensor, embedding: torch.Tensor, grid: list[torch.Tensor], **kwargs):
        for block in self.blocks:
            x = block(x, embedding, self.rope, grid)
        return x

class DiT1d(DiTGeneric):
    @beartype
    def __init__(self, patch_size: int|tuple[int, ...], num_channels: int, dim: int=768,
                 num_layers: int=12, num_heads: int=12, conditioners: dict[str, nn.Module]|None=None):
        """
        1D Diffusion Transformer.

        Args:
            patch_size: Size of the patches (int or 1-tuple).
            num_channels: Number of input channels.
            dim: Embedding dimension.
            num_layers: Number of DiT blocks.
            num_heads: Number of attention heads.
            conditioners: Dictionary of conditioner modules.
                - 'time': A module with key 'time' can be inserted to replace the default sinusoidal time embedding.
                - 'resolution': If 'resolution' appears in conditioners, a default resolution embedding module is applied regardless of its value.
                - Others: If other conditioners are inserted, they must project the corresponding input to (batch_size, dim). Ensure the parameter with the same name is provided in the model's forward method.
                - Note: 'context' and 'mask' are reserved names and should not be used in conditioners.
                If conditioners is None, a basic sinusoidal time embedding is applied and the input time t dtype must be float instead of int64.
        """
        super().__init__(patch_size, num_channels, dim, num_layers, num_heads, conditioners, ndim=1)

class DiT2d(DiTGeneric):
    @beartype
    def __init__(self, patch_size: int|tuple[int, ...], num_channels: int, dim: int=768,
                 num_layers: int=12, num_heads: int=12, conditioners: dict[str, nn.Module]|None=None):
        """
        2D Diffusion Transformer.

        Args:
            patch_size: Size of the patches (int or 2-tuple).
            num_channels: Number of input channels.
            dim: Embedding dimension.
            num_layers: Number of DiT blocks.
            num_heads: Number of attention heads.
            conditioners: Dictionary of conditioner modules.
                - 'time': A module with key 'time' can be inserted to replace the default sinusoidal time embedding.
                - 'resolution': If 'resolution' appears in conditioners, a default resolution embedding module is applied regardless of its value.
                - Others: If other conditioners are inserted, they must project the corresponding input to (batch_size, dim). Ensure the parameter with the same name is provided in the model's forward method.
                - Note: 'context' and 'mask' are reserved names and should not be used in conditioners.
                If conditioners is None, a basic sinusoidal time embedding is applied and the input time t dtype must be float instead of int64.
        """
        super().__init__(patch_size, num_channels, dim, num_layers, num_heads, conditioners, ndim=2)

class DiT3d(DiTGeneric):
    @beartype
    def __init__(self, patch_size: int|tuple[int, ...], num_channels: int, dim: int=768,
                 num_layers: int=12, num_heads: int=12, conditioners: dict[str, nn.Module]|None=None):
        """
        3D Diffusion Transformer.

        Args:
            patch_size: Size of the patches (int or 3-tuple).
            num_channels: Number of input channels.
            dim: Embedding dimension.
            num_layers: Number of DiT blocks.
            num_heads: Number of attention heads.
            conditioners: Dictionary of conditioner modules.
                - 'time': A module with key 'time' can be inserted to replace the default sinusoidal time embedding.
                - 'resolution': If 'resolution' appears in conditioners, a default resolution embedding module is applied regardless of its value.
                - Others: If other conditioners are inserted, they must project the corresponding input to (batch_size, dim). Ensure the parameter with the same name is provided in the model's forward method.
                - Note: 'context' and 'mask' are reserved names and should not be used in conditioners.
                If conditioners is None, a basic sinusoidal time embedding is applied and the input time t dtype must be float instead of int64.
        """
        super().__init__(patch_size, num_channels, dim, num_layers, num_heads, conditioners, ndim=3)

class ContextDiTGeneric(DiTBaseGeneric):
    @beartype
    def __init__(self, patch_size: int|tuple[int, ...], num_channels: int, dim: int,
                 num_layers: int, num_heads: int, conditioners: dict[str, nn.Module]|None, ndim: int):
        super().__init__(patch_size, num_channels, dim, num_layers, num_heads, conditioners, ndim)

        # DiT Blocks
        self.blocks = nn.ModuleList(
            [architecture.DiTBlockWithCrossAttention(dim, num_heads) for _ in range(num_layers)]
        )
    
    def _forward_blocks(self, x: torch.Tensor, embedding: torch.Tensor, grid: list[torch.Tensor], **kwargs):
        # Get necessary args
        assert 'context' in kwargs, 'context is required'
        context = kwargs['context']
        mask = kwargs.get('mask', None)
        # Blocks
        for block in self.blocks:
            x = block(x, embedding, context, self.rope, grid, mask)
        return x

class ContextDiT1d(ContextDiTGeneric):
    @beartype
    def __init__(self, patch_size: int|tuple[int, ...], num_channels: int, dim: int=768,
                 num_layers: int=12, num_heads: int=12, conditioners: dict[str, nn.Module]|None=None):
        """
        1D Contextual Diffusion Transformer. This model includes cross-attention layers for conditioning on external context.

        Args:
            patch_size: Size of the patches (int or 1-tuple).
            num_channels: Number of input channels.
            dim: Embedding dimension.
            num_layers: Number of DiT blocks.
            num_heads: Number of attention heads.
            conditioners: Dictionary of conditioner modules.
                - 'time': A module with key 'time' can be inserted to replace the default sinusoidal time embedding.
                - 'resolution': If 'resolution' appears in conditioners, a default resolution embedding module is applied regardless of its value.
                - Others: If other conditioners are inserted, they must project the corresponding input to (batch_size, dim). Ensure the parameter with the same name is provided in the model's forward method.
                - Note: 'context' and 'mask' are reserved names and should not be used in conditioners.
                If conditioners is None, a basic sinusoidal time embedding is applied and the input time t dtype must be float instead of int64.

        Forward Args:
            context: Context sequence tensor for cross-attention (required). Must have the same dimension as `dim`.
            mask: Attention mask for context (optional).
        """
        super().__init__(patch_size, num_channels, dim, num_layers, num_heads, conditioners, ndim=1)

class ContextDiT2d(ContextDiTGeneric):
    @beartype
    def __init__(self, patch_size: int|tuple[int, ...], num_channels: int, dim: int=768,
                 num_layers: int=12, num_heads: int=12, conditioners: dict[str, nn.Module]|None=None):
        """
        2D Contextual Diffusion Transformer. This model includes cross-attention layers for conditioning on external context.

        Args:
            patch_size: Size of the patches (int or 2-tuple).
            num_channels: Number of input channels.
            dim: Embedding dimension.
            num_layers: Number of DiT blocks.
            num_heads: Number of attention heads.
            conditioners: Dictionary of conditioner modules.
                - 'time': A module with key 'time' can be inserted to replace the default sinusoidal time embedding.
                - 'resolution': If 'resolution' appears in conditioners, a default resolution embedding module is applied regardless of its value.
                - Others: If other conditioners are inserted, they must project the corresponding input to (batch_size, dim). Ensure the parameter with the same name is provided in the model's forward method.
                - Note: 'context' and 'mask' are reserved names and should not be used in conditioners.
                If conditioners is None, a basic sinusoidal time embedding is applied and the input time t dtype must be float instead of int64.

        Forward Args:
            context: Context sequence tensor for cross-attention (required). Must have the same dimension as `dim`.
            mask: Attention mask for context (optional).
        """
        super().__init__(patch_size, num_channels, dim, num_layers, num_heads, conditioners, ndim=2)

class ContextDiT3d(ContextDiTGeneric):
    @beartype
    def __init__(self, patch_size: int|tuple[int, ...], num_channels: int, dim: int=768,
                 num_layers: int=12, num_heads: int=12, conditioners: dict[str, nn.Module]|None=None):
        """
        3D Contextual Diffusion Transformer. This model includes cross-attention layers for conditioning on external context.

        Args:
            patch_size: Size of the patches (int or 3-tuple).
            num_channels: Number of input channels.
            dim: Embedding dimension.
            num_layers: Number of DiT blocks.
            num_heads: Number of attention heads.
            conditioners: Dictionary of conditioner modules.
                - 'time': A module with key 'time' can be inserted to replace the default sinusoidal time embedding.
                - 'resolution': If 'resolution' appears in conditioners, a default resolution embedding module is applied regardless of its value.
                - Others: If other conditioners are inserted, they must project the corresponding input to (batch_size, dim). Ensure the parameter with the same name is provided in the model's forward method.
                - Note: 'context' and 'mask' are reserved names and should not be used in conditioners.
                If conditioners is None, a basic sinusoidal time embedding is applied and the input time t dtype must be float instead of int64.

        Forward Args:
            context: Context sequence tensor for cross-attention (required). Must have the same dimension as `dim`.
            mask: Attention mask for context (optional).
        """
        super().__init__(patch_size, num_channels, dim, num_layers, num_heads, conditioners, ndim=3)
