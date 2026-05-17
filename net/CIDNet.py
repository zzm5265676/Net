import torch
import torch.nn as nn
from net.HVI_transform import RGB_HVI
from net.transformer_utils import *
from net.LCA import *
from huggingface_hub import PyTorchModelHubMixin

class CIDNet(nn.Module, PyTorchModelHubMixin):
    def __init__(self, 
                 channels=[36, 36, 72, 144],
                 heads=[1, 2, 4, 8],
                 norm=False
        ):
        super(CIDNet, self).__init__()
        
        
        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads
        
        # HV_ways
        self.HVE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, ch1, 3, stride=1, padding=0,bias=False)
            )
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm = norm)
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm = norm)
        self.HVE_block3 = NormDownsample(ch3, ch4, use_norm = norm)
        
        self.HVD_block3 = NormUpsample(ch4, ch3, use_norm = norm)
        self.HVD_block2 = NormUpsample(ch3, ch2, use_norm = norm)
        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm = norm)
        self.HVD_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 2, 3, stride=1, padding=0,bias=False)
        )
        
        
        # I_ways
        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, ch1, 3, stride=1, padding=0,bias=False),
            )
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm = norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm = norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm = norm)
        
        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.ID_block0 =  nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 1, 3, stride=1, padding=0,bias=False),
            )
        
        self.HV_LCA1 = HV_LCA(ch2, head2)
        self.HV_LCA2 = HV_LCA(ch3, head3)
        self.HV_LCA3 = HV_LCA(ch4, head4)
        self.HV_LCA4 = HV_LCA(ch4, head4)
        self.HV_LCA5 = HV_LCA(ch3, head3)
        self.HV_LCA6 = HV_LCA(ch2, head2)
        
        self.I_LCA1 = I_LCA(ch2, head2)
        self.I_LCA2 = I_LCA(ch3, head3)
        self.I_LCA3 = I_LCA(ch4, head4)
        self.I_LCA4 = I_LCA(ch4, head4)
        self.I_LCA5 = I_LCA(ch3, head3)
        self.I_LCA6 = I_LCA(ch2, head2)
        
        self.trans = RGB_HVI()
        
    def forward(self, x):
        # ==========================================
        # 【步骤 1：准备工作 + RGB → HVI】
        # ==========================================
        dtypes = x.dtype  # 数据类型（不影响shape）
        hvi = self.trans.HVIT(x) # RGB 转 HVI
        i = hvi[:,2,:,:].unsqueeze(1).to(dtypes) # 抽取亮度通道 I

        # ==========================================
        # 【步骤 2：初始编码 + 下采样第一层】
        # ==========================================
        i_enc0 = self.IE_block0(i) # I 分支初始卷积 [B, 36, H, W]
        i_enc1 = self.IE_block1(i_enc0) # I 分支下采样 [B, 36, H/2, W/2]
        hv_0 = self.HVE_block0(hvi) # HV 分支初始卷积 [B, 36, H, W]
        hv_1 = self.HVE_block1(hv_0) # HV 分支下采样 [B, 36, H/2, W/2]
        # 跳跃连接保存
        i_jump0 = i_enc0
        hv_jump0 = hv_0

        # ==========================================
        # 【步骤 3：第一次交叉注意力 LCA1】
        # ==========================================
        i_enc2 = self.I_LCA1(i_enc1, hv_1) #[B, 36, H/2, W/2]
        hv_2 = self.HV_LCA1(hv_1, i_enc1) #[B, 36, H/2, W/2]
        v_jump1 = i_enc2
        hv_jump1 = hv_2
        i_enc2 = self.IE_block2(i_enc2) # 下采样 [B, 72, H/4, W/4]
        hv_2 = self.HVE_block2(hv_2) # 下采样 [B, 72, H/4, W/4]

        # ==========================================
        # 【步骤 4：第二次交叉注意力 LCA2】
        # ==========================================
        i_enc3 = self.I_LCA2(i_enc2, hv_2) #[B, 72, H/4, W/4]
        hv_3 = self.HV_LCA2(hv_2, i_enc2) #[B, 72, H/4, W/4]
        v_jump2 = i_enc3
        hv_jump2 = hv_3
        i_enc3 = self.IE_block3(i_enc2) #i_enc3 (下采样): [B, 144, H/8, W/8]
        hv_3 = self.HVE_block3(hv_2) #hv_3 (下采样): [B, 144, H/8, W/8]

        # ==========================================
        # 【步骤 5：最底层 第三次交叉注意力 LCA3】
        # ==========================================
        i_enc4 = self.I_LCA3(i_enc3, hv_3) #i_enc4: [B, 144, H/8, W/8]
        hv_4 = self.HV_LCA3(hv_3, i_enc3) #hv_4: [B, 144, H/8, W/8]
        
        # ==========================================
        # 【步骤 6：最底层 第四次交叉注意力 LCA4】
        # ==========================================
        i_dec4 = self.I_LCA4(i_enc4,hv_4) #i_dec4: [B, 144, H/8, W/8]
        hv_4 = self.HV_LCA4(hv_4, i_enc4) #hv_4: [B, 144, H/8, W/8]
        
        # ==========================================
        # 【步骤 7：解码器上采样 + 跳跃连接】
        # ==========================================
        # 上采样: [B, 144, H/8, W/8] → [B, 72, H/4, W/4] 拼接跳跃连接: [B, 72+72, H/4, W/4] → [B, 72, H/4, W/4]
        # [B, 72, H/4, W/4]
        hv_3 = self.HVD_block3(hv_4, hv_jump2)
        i_dec3 = self.ID_block3(i_dec4, v_jump2)

        # ==========================================
        # 【步骤 8：第五次交叉注意力 LCA5】
        # ==========================================
        i_dec2 = self.I_LCA5(i_dec3, hv_3) #i_dec2: [B, 72, H/4, W/4]
        hv_2 = self.HV_LCA5(hv_3, i_dec3) #hv_2: [B, 72, H/4, W/4]
        
        # ==========================================
        # 【步骤 9：继续上采样】
        # ==========================================
        # 上采样后 shape: [B, 36, H/2, W/2]
        hv_2 = self.HVD_block2(hv_2, hv_jump1)
        i_dec2 = self.ID_block2(i_dec3, v_jump1)
        
        # ==========================================
        # 【步骤 10：第六次交叉注意力 LCA6】
        # ==========================================
        i_dec1 = self.I_LCA6(i_dec2, hv_2) #i_dec1: [B, 36, H/2, W/2]
        hv_1 = self.HV_LCA6(hv_2, i_dec2) #hv_1: [B, 36, H/2, W/2]
        
        # ==========================================
        # 【步骤 11：最终上采样回到原图尺寸】
        # ==========================================
        i_dec1 = self.ID_block1(i_dec1, i_jump0) #i_dec1: [B, 36, H, W]
        i_dec0 = self.ID_block0(i_dec1) #i_dec0: [B, 1, H, W] (输出亮度 I)
        hv_1 = self.HVD_block1(hv_1, hv_jump0) #hv_1: [B, 36, H, W]
        hv_0 = self.HVD_block0(hv_1) #hv_0: [B, 2, H, W] (输出颜色 HV)
        
        # ==========================================
        # 【步骤 12：拼接 + 残差 + 逆变换回 RGB】
        # ==========================================
        # cat([HV, I]): [B, 2+1, H, W] = [B, 3, H, W]
        # 残差 + hvi: [B, 3, H, W]
        output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi
        # output_rgb: [B, 3, H, W] (最终增强图)
        output_rgb = self.trans.PHVIT(output_hvi)

        return output_rgb
    
    def HVIT(self,x):
        hvi = self.trans.HVIT(x)
        return hvi
    
    

