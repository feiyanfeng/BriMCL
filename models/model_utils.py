import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class PositionalEmbedding(nn.Module):
    __doc__ = r"""Computes a positional embedding of timesteps.

    Input:
        x: tensor of shape (N)
    Output:
        tensor of shape (N, dim)
    Args:
        dim (int): embedding dimension
        scale (float): linear scale to be applied to timesteps. Default: 1.0
    """

    def __init__(self, dim, scale=1.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale
        
        # half_dim = self.dim // 2
        # emb = math.log(10000) / half_dim

        # print(half_dim, emb, '*'*20)
        # self.emb = torch.exp(torch.arange(half_dim) * -emb)
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.outer(x * self.scale, emb)
        # emb = torch.outer(x * self.scale, self.emb.to(x.device))
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Residual_block(nn.Module):
    def __init__(self, in_ch, out_ch, num_groups):
        super().__init__()
        def block(in_ch, out_ch):
            return nn.Sequential(
                nn.GroupNorm(num_groups, in_ch),
                nn.SiLU(),
                nn.Conv3d(in_ch, out_ch, 3, padding=1)
            )
        self.block1 = block(in_ch, out_ch)
        self.drop = nn.Dropout(0.1)
        self.block2 = block(out_ch, out_ch)

    def forward(self, x):
        out = self.block1(x)
        return self.drop(self.block2(out)) + out


class Down_block(nn.Module):
    def __init__(self, in_ch, out_ch, base_channels, num_groups, time_emb_scale=1, use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        # 1.
        self.time_mlp = nn.Sequential(
            PositionalEmbedding(base_channels, time_emb_scale),
            nn.Linear(base_channels, in_ch),
            nn.SiLU(),
            nn.Linear(in_ch, in_ch)
        )
        # 2.
        self.residual_block = Residual_block( in_ch, out_ch, num_groups)
        # # 3.
        # if use_attention:
        #     self.attention = AttentionBlock(out_ch, num_groups=num_groups)
        # 4.
        self.downsample = nn.Conv3d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, t):
        # print('*'*10, 'Down_block', '*'*10)
        if t is not None:
            x = x + self.time_mlp(t)[:, :, None, None, None]
        # residual_block
        out = self.residual_block(x)
        # # attention
        # if self.use_attention:
        #     out = self.attention(out)
        # downsample
        out2 = self.downsample(out)
        return out, out2  # 采样前， 采样后


class AttentionBlock(nn.Module):
    __doc__ = r"""Applies QKV self-attention with a residual connection.
    
    Input:
        x: tensor of shape (N, in_channels, H, W)
        norm (string or None): which normalization to use (instance, group, batch, or none). Default: "gn"
        num_groups (int): number of groups used in group normalization. Default: 32
    Output:
        tensor of shape (N, in_channels, H, W)
    Args:
        in_channels (int): number of input channels
    """
    def __init__(self, in_channels, num_groups):
        super().__init__()
        
        self.in_channels = in_channels
        self.norm = nn.GroupNorm(num_groups, in_channels)  # TODO: 在源代码处是可选的
        self.to_qkv = nn.Conv3d(in_channels, in_channels * 3, 1)
        self.to_out = nn.Conv3d(in_channels, in_channels, 1)

    def forward(self, x):
        b, c, d, h, w = x.shape
        q, k, v = torch.split(self.to_qkv(self.norm(x)), self.in_channels, dim=1)

        q = q.permute(0, 2, 3, 4, 1).view(b, d * h * w, c)
        k = k.view(b, c, d * h * w)
        v = v.permute(0, 2, 3, 4, 1).view(b, d * h * w, c)

        dot_products = torch.bmm(q, k) * (c ** (-0.5))
        assert dot_products.shape == (b, d * h * w, d * h * w)

        attention = torch.softmax(dot_products, dim=-1)
        out = torch.bmm(attention, v)
        assert out.shape == (b, d * h * w, c)
        out = out.view(b, d, h, w, c).permute(0, 4, 1, 2, 3)

        return self.to_out(out) + x


class Bottleneck(nn.Module):
    def __init__(self, ch, num_groups):
        super().__init__()
        self.residual_block1 = Residual_block(ch, ch, num_groups)
        self.attention = AttentionBlock(ch,num_groups=num_groups)
        self.residual_block2 = Residual_block(ch, ch, num_groups)
    def forward(self, x):
        out = self.residual_block1(x)
        out = self.attention(out)
        return self.residual_block2(out)



class Up_block(nn.Module):
    def __init__(self, in_ch, out_ch, base_channels, time_emb_scale, num_groups, use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        # 1.
        self.time_mlp = nn.Sequential(
            PositionalEmbedding(base_channels, time_emb_scale),
            nn.Linear(base_channels, in_ch),
            nn.SiLU(),
            nn.Linear(in_ch, in_ch)
        )
        # 2.
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv3d(in_ch, in_ch, 3, padding=1),
        )
        # 3.
        self.residual_block = Residual_block(in_ch*2, out_ch, num_groups)
        # 4.
        if use_attention:
            self.attention = AttentionBlock(out_ch, num_groups=num_groups)

    def forward(self, x, x_, t):
        # x: x_下采样，经过一系列操作得到的

        # add time
        if t is not None:
            x = x + self.time_mlp(t)[:, :, None, None, None]
        # upsample
        x = self.upsample(x)
        # connect
        x = torch.cat([x, x_], dim=1)
        # residual block  
        out = self.residual_block(x)
        # attention  TODO: attention和上/下采样谁前谁后
        if self.use_attention:
            out = self.attention(out)

        return out  # TODO: 是否需要sigmoid
