"""
YOLO11 注意力模块：残差 CoordAtt + 残差 ECA
残差形式保证初始恒等映射，不破坏预训练特征
"""
import torch
import torch.nn as nn


# ============================================================================
# Coordinate Attention 残差版 (CVPR 2021)
# x + alpha * x * attn_h * attn_w, alpha 初始 0.01
# ============================================================================
class CoordAtt(nn.Module):
    def __init__(self, c1, c2=None, reduction=32):
        super().__init__()
        if c2 is None:
            c2 = c1
        mip = max(8, c1 // reduction)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(c1, mip, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()

        self.conv_h = nn.Conv2d(mip, c2, kernel_size=1, bias=True)
        self.conv_w = nn.Conv2d(mip, c2, kernel_size=1, bias=True)

        # gate 初始≈0 (bias=-3, sigmoid≈0.047), alpha=1.0, 初始贡献≈0.2%
        # 训练后 gate 可自由增长，实际增益范围 [1.0, 2.0]
        nn.init.constant_(self.conv_h.bias, -3.0)
        nn.init.constant_(self.conv_w.bias, -3.0)
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        _, _, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        gate = self.conv_h(x_h).sigmoid() * self.conv_w(x_w).sigmoid()
        return x + self.alpha * x * gate


# ============================================================================
# ECA 残差版 (CVPR 2020)
# x + alpha * x * channel_attn, alpha 初始 0.01
# ============================================================================
class ECA(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=True)
        self.sigmoid = nn.Sigmoid()
        nn.init.constant_(self.conv.bias, -3.0)
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x + self.alpha * x * self.sigmoid(y)
