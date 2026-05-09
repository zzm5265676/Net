import torch
import torch.nn as nn

pi = 3.141592653589793

class RGB_HVI(nn.Module):
    def __init__(self, k_min=0.05, k_max=0.80):
        super(RGB_HVI, self).__init__()

        # 空间自适应 k(x) 的范围
        self.k_min = k_min
        self.k_max = k_max

        # 用一个轻量 CNN 预测 k_map
        # 输入 RGB，输出 [B,1,H,W]
        self.k_predictor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        # 原代码中的控制参数，先保留接口
        self.gated = False
        self.gated2 = False
        self.alpha = 1.0
        self.alpha_s = 1.3

    def _rgb_to_hsv_components(self, img):
        eps = 1e-8
        device = img.device
        dtype = img.dtype

        # Imax 和 Imin
        value = img.max(dim=1)[0].to(dtype)      # [B,H,W]
        img_min = img.min(dim=1)[0].to(dtype)    # [B,H,W]

        hue = torch.empty(
            img.shape[0], img.shape[2], img.shape[3],
            device=device, dtype=dtype
        )

        delta = value - img_min

        # RGB 通道
        r = img[:, 0]
        g = img[:, 1]
        b = img[:, 2]

        # Hue 分段计算
        mask_b = (b == value)
        mask_g = (g == value)
        mask_r = (r == value)

        hue[mask_b] = 4.0 + ((r - g) / (delta + eps))[mask_b]
        hue[mask_g] = 2.0 + ((b - r) / (delta + eps))[mask_g]
        hue[mask_r] = (((g - b) / (delta + eps))[mask_r]) % 6.0

        # 灰度点：max == min，色相设为 0
        hue[delta == 0] = 0.0

        # 归一化到 [0,1)
        hue = hue / 6.0

        # Saturation
        saturation = delta / (value + eps)
        saturation[value == 0] = 0.0

        return hue.unsqueeze(1), saturation.unsqueeze(1), value.unsqueeze(1)

    def _predict_k_map(self, img):

        k_raw = self.k_predictor(img)  # [B,1,H,W], range [0,1]
        k_map = self.k_min + (self.k_max - self.k_min) * k_raw
        return k_map

    def HVIT(self, img, return_aux=False, aux=None):

        eps = 1e-8

        hue, saturation, value = self._rgb_to_hsv_components(img)

        # -------------------------------------------------
        # 关键修改：
        # 如果 aux 里已经有 k_map，就使用外部传入的 k_map；
        # 否则才根据当前 img 重新预测 k_map。
        # -------------------------------------------------
        if aux is not None and "k_map" in aux:
            k_map = aux["k_map"]

            # 防止极少数情况下尺寸不一致
            if k_map.shape[-2:] != value.shape[-2:]:
                k_map = torch.nn.functional.interpolate(
                    k_map,
                    size=value.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )
        else:
            k_map = self._predict_k_map(img)

        # C_{k(x)}(x)
        base = torch.sin(value * 0.5 * pi) + eps
        color_sensitive = base.pow(k_map)

        # Hue 极化
        ch = torch.cos(2.0 * pi * hue)
        cv = torch.sin(2.0 * pi * hue)

        # HVI 三通道
        H = color_sensitive * saturation * ch
        V = color_sensitive * saturation * cv
        I = value

        hvi = torch.cat([H, V, I], dim=1)

        if return_aux:
            aux_out = {
                "k_map": k_map
            }
            return hvi, aux_out

        return hvi

    def PHVIT(self, img, aux=None):

        eps = 1e-8

        H = img[:, 0:1, :, :]
        V = img[:, 1:2, :, :]
        I = img[:, 2:3, :, :]

        # 数值范围限制
        H = torch.clamp(H, -1.0, 1.0)
        V = torch.clamp(V, -1.0, 1.0)
        I = torch.clamp(I, 0.0, 1.0)

        if aux is None or "k_map" not in aux:
            raise ValueError("PHVIT needs aux['k_map'] when using adaptive k(x). ""Please call HVIT(x, return_aux=True) in forward first.")

        k_map = aux["k_map"]

        # 用输出 I 和同一个 k_map 计算逆变换需要的 C
        base = torch.sin(I * 0.5 * pi) + eps
        color_sensitive = base.pow(k_map)

        H = H / (color_sensitive + eps)
        V = V / (color_sensitive + eps)

        H = torch.clamp(H, -1.0, 1.0)
        V = torch.clamp(V, -1.0, 1.0)

        # 恢复 HSV
        h = torch.atan2(V + eps, H + eps) / (2.0 * pi)
        h = h % 1.0

        s = torch.sqrt(H ** 2 + V ** 2 + eps)

        if self.gated:
            s = s * self.alpha_s

        s = torch.clamp(s, 0.0, 1.0)
        v = torch.clamp(I, 0.0, 1.0)

        # HSV -> RGB
        hi = torch.floor(h * 6.0)
        f = h * 6.0 - hi

        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)

        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)

        hi0 = hi == 0
        hi1 = hi == 1
        hi2 = hi == 2
        hi3 = hi == 3
        hi4 = hi == 4
        hi5 = hi == 5

        r[hi0] = v[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]

        r[hi1] = q[hi1]
        g[hi1] = v[hi1]
        b[hi1] = p[hi1]

        r[hi2] = p[hi2]
        g[hi2] = v[hi2]
        b[hi2] = t[hi2]

        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = v[hi3]

        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = v[hi4]

        r[hi5] = v[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]

        rgb = torch.cat([r, g, b], dim=1)

        if self.gated2:
            rgb = rgb * self.alpha

        return torch.clamp(rgb, 0.0, 1.0)