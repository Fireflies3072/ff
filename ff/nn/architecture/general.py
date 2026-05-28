import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention2d(nn.Module):
    """
    Self-attention block for 2D images. Compatible with Flash Attention.
    """
    def __init__(self, dim):
        super().__init__()
        self.conv_query = nn.Conv2d(dim, dim // 8, 1)
        self.conv_key = nn.Conv2d(dim, dim // 8, 1)
        self.conv_value = nn.Conv2d(dim, dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x, condition=None):
        b, c, h, w = x.shape
        query = self.conv_query(x).view(b, 1, c // 8, h * w).permute(0, 1, 3, 2).contiguous() # (B, 1, L, D_q)
        key = self.conv_key(x).view(b, 1, c // 8, h * w).permute(0, 1, 3, 2).contiguous() # (B, 1, L, D_k)
        value = self.conv_value(x).view(b, 1, c, h * w).permute(0, 1, 3, 2).contiguous() # (B, 1, L, D_v)
        attention = F.scaled_dot_product_attention(query, key, value) # (B, 1, L, D_v)
        attention = attention.permute(0, 1, 3, 2).view(b, c, h, w)
        return x + self.gamma * attention

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, scale='same', groups=8):
        super().__init__()
        # Check if the number of groups is correct
        in_groups = groups if in_dim % groups == 0 else 1
        out_groups = groups if out_dim % groups == 0 else 1
        
        self.conv1 = nn.Sequential(
            nn.GroupNorm(in_groups, in_dim), 
            nn.SiLU(),
            nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        )
        self.conv2_pre = nn.Sequential(
            nn.GroupNorm(out_groups, out_dim),
            nn.SiLU()
        )
        if scale == 'same':
            self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)
            self.shortcut = nn.Conv2d(in_dim, out_dim, 1, 1, 0) if in_dim != out_dim else nn.Identity()
        elif scale == 'down':
            self.conv2 = nn.Conv2d(out_dim, out_dim, 4, 2, 1)
            shortcut_layers = [nn.AvgPool2d(2, 2)]
            if in_dim != out_dim:
                shortcut_layers.insert(0, nn.Conv2d(in_dim, out_dim, 1, 1, 0))
            self.shortcut = nn.Sequential(*shortcut_layers)
        elif scale == 'up':
            self.conv2 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(out_dim, out_dim, 3, 1, 1)
            )
            shortcut_layers = [nn.Upsample(scale_factor=2, mode='nearest')]
            if in_dim != out_dim:
                shortcut_layers.insert(0, nn.Conv2d(in_dim, out_dim, 1, 1, 0))
            self.shortcut = nn.Sequential(*shortcut_layers)
        else:
            raise ValueError(f"Invalid scale: {scale}")

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2_pre(h)
        h = self.conv2(h)
        return h + self.shortcut(x)

class ResidualBlockWithEmbedding(ResidualBlock):
    def __init__(self, in_dim, out_dim, embedding_dim, scale='same', groups=8):
        super().__init__(in_dim, out_dim, scale, groups)
        
        self.embedding_mlp = nn.Linear(embedding_dim, out_dim)

    def forward(self, x, embedding):
        h = self.conv1(x)
        h = h + self.embedding_mlp(embedding)[:, :, None, None]
        h = self.conv2_pre(h)
        h = self.conv2(h)
        return h + self.shortcut(x)
