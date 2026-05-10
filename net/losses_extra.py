import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelEdgeLoss(nn.Module):
    """
    边缘保持损失:
    L_edge = || Sobel(pred) - Sobel(gt) ||_1
    """
    def __init__(self):
        super().__init__()

        gx = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        gy = torch.tensor(
            [[-1, -2, -1],
             [ 0,  0,  0],
             [ 1,  2,  1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("gx", gx)
        self.register_buffer("gy", gy)

    def gradient(self, x):
        """
        x: [B, C, H, W]
        """
        b, c, h, w = x.shape

        gx = self.gx.to(device=x.device, dtype=x.dtype).repeat(c, 1, 1, 1)
        gy = self.gy.to(device=x.device, dtype=x.dtype).repeat(c, 1, 1, 1)

        grad_x = F.conv2d(x, gx, padding=1, groups=c)
        grad_y = F.conv2d(x, gy, padding=1, groups=c)

        return torch.abs(grad_x) + torch.abs(grad_y)

    def forward(self, pred, target):
        pred_grad = self.gradient(pred)
        target_grad = self.gradient(target)
        return F.l1_loss(pred_grad, target_grad)


def dark_weighted_l1(pred, target, low, alpha=2.0, eps=1e-6):
    """
    暗区加权 L1。

    pred:   output_rgb, [B,3,H,W]
    target: gt_rgb,     [B,3,H,W]
    low:    input low,  [B,3,H,W]

    w(x) = 1 + alpha * (1 - I_low(x))
    """
    with torch.no_grad():
        i_low = low.max(dim=1, keepdim=True)[0]   # [B,1,H,W]
        weight = 1.0 + alpha * (1.0 - i_low)

    loss = torch.abs(pred - target) * weight

    # pred 有 3 个通道，所以分母乘 pred.shape[1]
    return loss.sum() / (weight.sum() * pred.shape[1] + eps)


def color_ratio_loss(pred, target, eps=1e-6):
    """
    颜色比例一致性损失。
    用 log ratio 比直接 ratio 更稳定。

    约束:
    log(R/G), log(R/B), log(G/B)
    """
    pred_mean = pred.mean(dim=[2, 3])      # [B,3]
    target_mean = target.mean(dim=[2, 3])  # [B,3]

    pairs = [(0, 1), (0, 2), (1, 2)]
    loss = pred.new_tensor(0.0)

    for i, j in pairs:
        pred_ratio = torch.log((pred_mean[:, i] + eps) / (pred_mean[:, j] + eps))
        target_ratio = torch.log((target_mean[:, i] + eps) / (target_mean[:, j] + eps))
        loss = loss + F.l1_loss(pred_ratio, target_ratio)

    return loss