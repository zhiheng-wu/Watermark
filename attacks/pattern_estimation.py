import torch
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from .core import BaseAttacker, AttackerFactory

@AttackerFactory.register("pattern_estimation")
class PatternEstimationAttacker(BaseAttacker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.clean_path = self.params.get('clean_path')
        self.watermarked_path = self.params.get('watermarked_path')
        self.num_images = self.params.get('num_images', 100)
        self.init_size = self.params.get('image_size', (512, 512)) 
        self.factor = self.params.get('factor', 1.0)
        
        if not self.clean_path or not self.watermarked_path:
            raise ValueError("Requires 'clean_path' and 'watermarked_path'.")

        # 预计算模式 (H, W, C), 范围大约在 -255 到 255 之间
        self.estimated_pattern_np = self._compute_difference_pattern()

        # 缓存 Tensor 版本的 Pattern 以减少重复转换开销 (Lazy loading)
        self.pattern_tensor = None 

    def _sum_images(self, path, count):
        """读取并计算平均值"""
        image_sum = None
        processed_count = 0
        files = sorted([f for f in os.listdir(path) if f.lower().endswith(('png', 'jpg', 'jpeg'))])[:count]

        for file in tqdm(files, desc=f"Loading {os.path.basename(path)}"):
            img_path = os.path.join(path, file)
            # Resize 到统一尺寸进行计算
            image = Image.open(img_path).resize(self.init_size).convert('RGB')
            image_arr = np.array(image, dtype=np.float32)

            if image_sum is None:
                image_sum = image_arr
            else:
                image_sum += image_arr
            processed_count += 1
            
        return image_sum / processed_count if processed_count > 0 else None

    def _compute_difference_pattern(self):
        print(f"[Pattern] Estimating pattern from {self.num_images} images...")
        clean_avg = self._sum_images(self.clean_path, self.num_images)
        wm_avg = self._sum_images(self.watermarked_path, self.num_images)
        # 结果形状: (H, W, 3)
        return wm_avg - clean_avg

    def _prepare_pattern_tensor(self, target_device, target_shape):
        """
        将 NumPy Pattern 转换为适应当前输入的 Tensor
        target_shape: (B, C, H, W) 或 (C, H, W)
        """
        # 1. 转为 Tensor 并移至 GPU: (H, W, C) -> (C, H, W)
        if self.pattern_tensor is None or self.pattern_tensor.device != target_device:
            p = torch.from_numpy(self.estimated_pattern_np).to(target_device, dtype=torch.float32)
            self.pattern_tensor = p.permute(2, 0, 1) # (C, H, W)

        pattern = self.pattern_tensor
        
        # 2. 确定目标 H, W
        if len(target_shape) == 4: # (B, C, H, W)
            dst_h, dst_w = target_shape[2], target_shape[3]
        else: # (C, H, W)
            dst_h, dst_w = target_shape[1], target_shape[2]

        # 3. 如果尺寸不匹配，进行插值缩放
        if pattern.shape[1] != dst_h or pattern.shape[2] != dst_w:
            # interpolate 需要 (N, C, H, W) 维度
            pattern = F.interpolate(
                pattern.unsqueeze(0), 
                size=(dst_h, dst_w), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0) # 变回 (C, H, W)
            
        return pattern

    def attack(self, image: torch.Tensor) -> torch.Tensor:
        """
        支持 Batch 处理的攻击函数。
        输入: Tensor (B, C, H, W) 或 (C, H, W), 范围 [0, 1]
        """
        # 确保输入是 Tensor
        if not isinstance(image, torch.Tensor):
            raise TypeError("PatternEstimationAttacker now strictly expects torch.Tensor input for batch processing.")
        
        input_device = image.device
        input_ndim = image.dim()
        
        # 1. 准备 Pattern (C, H, W)，已经 Resize 好了，且在同一 Device 上
        # 注意：NumPy Pattern 是基于 0-255 计算的，而输入 Tensor 通常是 0-1
        # 所以我们需要将 Pattern 除以 255.0 来归一化
        pattern = self._prepare_pattern_tensor(input_device, image.shape) / 255.0
        
        # 2. 应用攻击 (Batch 减法)
        # 如果 image 是 (B, C, H, W)，pattern 是 (C, H, W)
        # PyTorch 会自动广播 pattern 到 (B, C, H, W)
        adv_image = image - (self.factor * pattern)
        
        # 3. 截断范围并保持原有维度
        adv_image = torch.clamp(adv_image, 0.0, 1.0)
        
        return adv_image