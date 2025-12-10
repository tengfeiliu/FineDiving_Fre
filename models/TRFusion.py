import torch
import torch.nn as nn

class TRFusionAttension(nn.Module):
    """
    将时域特征与频域特征进行融合，采用交叉注意力机制
    """
    def __init__(self, in_dim, out_dim=64, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(TRFusionAttension, self).__init__()

        self.num_heads = num_heads
        self.in_dim = in_dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_map = nn.Linear(in_dim, out_dim, bias=qkv_bias)
        self.k_map = nn.Linear(in_dim, out_dim, bias=qkv_bias)
        self.v_map = nn.Linear(in_dim, out_dim, bias=qkv_bias)
        self.proj = nn.Linear(64, out_dim)
        self.relu = nn.ReLU()

    def forward(self, q, v):
        B, N, C = q.shape
        C = C
        k = v
        NK = k.size(1)

        q = self.q_map(q)
        q = q.view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_map(k).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_map(v).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        output = self.proj(x)
        # output = torch.sigmoid(self.proj(x))# 增加了门控机制
        output = self.relu(output)
        return output