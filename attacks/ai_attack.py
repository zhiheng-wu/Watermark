
from models.attackers.CtrlRegen.ctrl import ctrl_regen_plus

import torch
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from .core import BaseAttacker, AttackerFactory

@AttackerFactory.register("ctrl_regen")
class CtrlRegenAttacker(BaseAttacker):
    """
    基于 ControlNet 重绘的新型攻击 (CtrlRegen)。
    
    参数 (params):
    - step: 控制重绘强度的参数 (对应原函数中的 strength/step)，默认 0.4。
    - seed: 随机种子，默认 0。
    
    功能：
    将输入的 Batch Tensor 转换为 PIL 图片，逐张调用 ctrl_regen_plus 进行攻击，
    并将结果拼接回 Batch Tensor。
    """
    def attack(self, image: torch.Tensor) -> torch.Tensor:
        step = self.params.get('step', 0.4)
        seed = self.params.get('seed', 42)
        # 1. 记录原始信息
        device = image.device
        original_dtype = image.dtype
        print
        # 处理输入维度，确保是 (B, C, H, W)
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        B, C, H, W = image.shape
        
        # 2. 准备输出容器
        # 由于 Stable Diffusion 输出通常是 512x512
        # 我们需要决定是返回 512 还是 resize 回原图。
        # 为了保证攻击前后的一致性，通常攻击者会把图片还原回原始分辨率。
        output_tensor_list = []

        # 3. 逐张处理 (Batch Loop)
        # 因为 ctrl_regen_plus 内部写死了处理单张图逻辑 (.images[0])
        for i in range(B):
            # --- 数据转换: GPU Tensor -> CPU PIL ---
            # 取出单张 (C, H, W)
            single_img_tensor = image[i]
            
            # 转换流程: Detach -> CPU -> Clamp(0,1) -> ToPIL
            # 这一步是为了适配 Diffusers Pipeline 对输入的期望
            pil_img = TF.to_pil_image(single_img_tensor.detach().cpu().clamp(0, 1))
            
            # --- 调用外部攻击函数 ---
            # 这一步会调用显存中的 SD 模型进行推理
            # 注意：ctrl_regen_plus 内部会将图片 resize 到 512x512
            attacked_pil = ctrl_regen_plus(pil_img, step=step, seed=seed)
            
            # --- 数据恢复: PIL -> GPU Tensor ---
            # 攻击后的图片是 PIL 格式，可能为 512x512
            
            # 如果需要保持输入输出尺寸严格一致，这里需要 Resize 回去
            if attacked_pil.size != (W, H):
                attacked_pil = attacked_pil.resize((W, H), resample=Image.BILINEAR)
            
            # 转回 Tensor 并归一化到 [0, 1]
            attacked_tensor = TF.to_tensor(attacked_pil)
            output_tensor_list.append(attacked_tensor)

        # 4. 拼接结果
        # Stack 后的形状: (B, C, H, W)
        result = torch.stack(output_tensor_list)
        
        # 5. 迁移回原始设备并恢复维度
        # 如果输入只有 3 维，输出也保持 3 维
        if B == 1 and image.shape != result.shape:
             result = result.squeeze(0)
             
        return result.to(device, dtype=original_dtype)