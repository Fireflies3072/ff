import torch
import torch.nn as nn
import torch.nn.functional as F

from ..architecture import ResidualBlockWithEmbedding, SelfAttention2d, SinusoidalEmbedding

class BaseUNet(nn.Module):
    def __init__(self, num_channels=3, dim=64, embedding_dim=256,
                 layers=(1, 2, 2, 2), num_mid_layers=1, T=1000, condition_dim=0):
        super().__init__()
        self.num_channels = num_channels
        self.dim = dim
        self.embedding_dim = embedding_dim
        self.layers = layers
        self.num_mid_layers = num_mid_layers
        self.T = T

        # Embedding layers
        self.time_embedding = SinusoidalEmbedding(embedding_dim, T)
        self.embedding_mlp = nn.Sequential(
            nn.Linear(embedding_dim + condition_dim, embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.SiLU()
        )

        # Input output layers
        self.conv_in = nn.Conv2d(num_channels, dim, 3, 1, 1)
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.Conv2d(dim, num_channels, 3, 1, 1)
        )

        # Mid blocks
        d = dim * (2 ** (len(layers) - 1))
        self.mid = nn.ModuleList()
        self.mid.append(ResidualBlockWithEmbedding(d, d, embedding_dim, scale='same'))
        self.mid.append(SelfAttention2d(d))
        for _ in range(num_mid_layers):
            self.mid.append(ResidualBlockWithEmbedding(d, d, embedding_dim, scale='same'))
        

        # UNet down blocks
        self.down_blocks = nn.ModuleList()
        for stage_index in range(len(layers)):
            num_blocks = layers[stage_index]
            out_dim = dim * (2 ** stage_index)
            stage_blocks = nn.ModuleList()
            for block_index in range(num_blocks):
                if stage_index > 0 and block_index == 0:
                    in_dim = dim * (2 ** (stage_index - 1))
                    scale_mode = 'down'
                else:
                    in_dim = out_dim
                    scale_mode = 'same'
                # Add residual block
                stage_blocks.append(ResidualBlockWithEmbedding(in_dim, out_dim, embedding_dim, scale=scale_mode))
                # Add self attention block
                if stage_index == 2:
                    stage_blocks.append(SelfAttention2d(out_dim))
            # Add down stage
            self.down_blocks.append(stage_blocks)
        
        # UNet up blocks
        self.up_blocks = nn.ModuleList()
        for stage_index in range(len(layers) - 1, -1, -1):
            num_blocks = layers[stage_index]
            ref_dim = dim * (2 ** stage_index)
            stage_blocks = nn.ModuleList()
            for block_index in range(num_blocks):
                if stage_index > 0 and block_index == num_blocks - 1:
                    in_dim = ref_dim if block_index > 0 else ref_dim * 2
                    out_dim = dim * (2 ** (stage_index - 1))
                    scale_mode = 'up'
                elif block_index == 0:
                    in_dim = ref_dim * 2
                    out_dim = ref_dim
                    scale_mode = 'same'
                else:
                    in_dim = ref_dim
                    out_dim = ref_dim
                    scale_mode = 'same'
                # Add residual block
                stage_blocks.append(ResidualBlockWithEmbedding(in_dim, out_dim, embedding_dim, scale=scale_mode))
                # Add self attention block
                if stage_index == 2:
                    stage_blocks.append(SelfAttention2d(out_dim))
            # Add up stage
            self.up_blocks.append(stage_blocks)

    def forward(self, x, t, condition=None):
        # Embedding
        embedding = self._get_embedding(t, condition)
        # Input
        h = self.conv_in(x)

        # Down blocks
        hidden_states = []
        for down_stage_blocks in self.down_blocks:
            for down_block in down_stage_blocks:
                h = down_block(h, embedding)
            hidden_states.append(h)
        
        # Mid blocks
        for mid_block in self.mid:
            h = mid_block(h, embedding)

        # Up blocks
        for up_stage_blocks, hidden_state in zip(self.up_blocks, reversed(hidden_states)):
            h = up_stage_blocks[0](torch.cat([h, hidden_state], dim=1), embedding)
            for up_block in up_stage_blocks[1:]:
                h = up_block(h, embedding)
        
        # Output
        return self.conv_out(h)
    
    def _get_embedding(self, t, condition=None):
        t_embedding = self.time_embedding(t)
        embedding = self.embedding_mlp(t_embedding)
        return embedding

class UnconditionalUNet(BaseUNet):
    def __init__(self, num_channels=3, dim=64, embedding_dim=256,
                 layers=(1, 2, 2, 2), num_mid_layers=1, T=1000):
        super().__init__(num_channels, dim, embedding_dim, layers, num_mid_layers, T)
    
    def forward(self, x, t):
        return super().forward(x, t)

class ClassConditionalUNet(BaseUNet):
    def __init__(self, num_channels=3, num_classes=10, dim=64, embedding_dim=256, layers=(1, 2, 2, 2), num_mid_layers=1, T=1000):
        super().__init__(num_channels, dim, embedding_dim, layers, num_mid_layers, T, embedding_dim)
        self.num_classes = num_classes
        self.label_embedding = nn.Embedding(num_classes + 1, embedding_dim)
    
    def forward(self, x, t, label):
        return super().forward(x, t, label)
    
    def _get_embedding(self, t, condition):
        t_embedding = self.time_embedding(t)
        l_embedding = self.label_embedding(condition)
        embedding = self.embedding_mlp(torch.cat([t_embedding, l_embedding], dim=1))
        return embedding
