import torch.nn.functional as F
import torchvision.transforms.functional as TF
from attacks.core import BaseAttacker, AttackerFactory
import torch

@AttackerFactory.register("scaling")
class ScalingAttacker(BaseAttacker):
    """
    GPU 上的缩放攻击。使用 F.interpolate 进行缩放，F.pad 进行补边。
    - 确保输入 image 格式为 (C, H, W) 或 (B, C, H, W) 且在 GPU 上。
    """
    def attack(self, image: torch.Tensor) -> torch.Tensor:
        scale = self.params.get('scale', 0.8)
        
        # 1. 自动添加 Batch 维度，如果缺失 (H, W, C) -> (1, C, H, W)
        if image.dim() == 3:
            C, H, W = image.shape
            img = image.unsqueeze(0) # (1, C, H, W)
        else:
            _, C, H, W = image.shape
            img = image

        # 2. 执行缩放 (F.interpolate)
        resized_img = F.interpolate(
            img, 
            scale_factor=scale, 
            mode='bilinear',
            align_corners=False
        )
        
        _, _, H_new, W_new = resized_img.shape

        # 3. 恢复到原始尺寸 (裁剪或补边)
        if scale < 1.0:
            # 缩小后，需要补边 (Padding) 到原图大小
            pad_h = H - H_new
            pad_w = W - W_new
            
            # F.pad 的顺序是 (left, right, top, bottom)
            padding = [
                pad_w // 2, pad_w - pad_w // 2,  # left, right
                pad_h // 2, pad_h - pad_h // 2   # top, bottom
            ]
            
            restored_img = F.pad(
                resized_img, 
                padding, 
                mode='constant', 
                value=0.0 # 补零 (黑色)
            ) # 仍然在 GPU 上
            
        elif scale > 1.0:
            # 放大后，需要裁剪 (Cropping) 到原图大小
            
            # 计算裁剪的起始点
            start_h = (H_new - H) // 2
            start_w = (W_new - W) // 2
            
            # 执行裁剪
            restored_img = resized_img[
                :, :, 
                start_h : start_h + H,
                start_w : start_w + W
            ] # 仍然在 GPU 上
            
        else:
            restored_img = resized_img
            
        # 4. 移除 Batch 维度（如果之前添加了）
        return restored_img.squeeze(0) if image.dim() == 3 else restored_img
    


@AttackerFactory.register("rotation")
class RotationAttacker(BaseAttacker):
    """
    GPU 上的旋转攻击。使用 TF.rotate (基于 grid_sample)。
    - 确保输入 image 格式为 (C, H, W) 或 (B, C, H, W) 且在 GPU 上。
    - TF.rotate 默认保持原始尺寸，并用 0 填充边界。
    """
    def attack(self, image: torch.Tensor) -> torch.Tensor:
        angle = self.params.get('angle', 5.0) # 顺时针为正
        
        # TF.rotate 要求输入是 (C, H, W) 或 (H, W, C)，且是 Tensor 或 PIL Image
        rotated_image = TF.rotate(
            image, 
            angle, 
            fill=0, # 用 0 (黑色) 填充新创建的区域
            interpolation=TF.InterpolationMode.BILINEAR # 确保在 GPU 上使用插值
        ) # 仍然在 GPU 上
            
        return rotated_image


@AttackerFactory.register("cropping")
class CroppingAttacker(BaseAttacker):
    """
    GPU 上的裁剪攻击。通过索引和 F.pad 实现。
    - 确保输入 image 格式为 (C, H, W) 或 (B, C, H, W) 且在 GPU 上。
    """
    def attack(self, image: torch.Tensor) -> torch.Tensor:
        crop_ratio = self.params.get('crop_ratio', 0.9)
        crop_start_h = self.params.get('crop_start_h', None)  # 新增：高度起始点
        crop_start_w = self.params.get('crop_start_w', None)  
        # 1. 自动添加 Batch 维度，如果缺失
        if image.dim() == 3:
            C, H, W = image.shape
            img = image.unsqueeze(0) # (1, C, H, W)
        else:
            _, C, H, W = image.shape
            img = image
        
        # 2. 计算裁剪尺寸和起始点
        H_crop, W_crop = int(H * crop_ratio), int(W * crop_ratio)

        if crop_start_h is None:
            start_h = (H - H_crop) // 2  # 默认中心裁剪
        else:
            start_h = max(0, min(crop_start_h, H - H_crop))
        
        if crop_start_w is None:
            start_w = (W - W_crop) // 2  # 默认中心裁剪
        else:
            start_w = max(0, min(crop_start_w, W - W_crop))
        
        # 3. 执行裁剪 (GPU 上的张量索引操作)
        cropped_img = img[
            :, :, 
            start_h : start_h + H_crop,
            start_w : start_w + W_crop
        ] # 仍然在 GPU 上
        
        # 4. 恢复到原始尺寸 (补边)
        pad_h = H - H_crop
        pad_w = W - W_crop
        
        # F.pad 的顺序是 (left, right, top, bottom)
        padding = [
            pad_w // 2, pad_w - pad_w // 2,
            pad_h // 2, pad_h - pad_h // 2
        ]
        
        restored_img = F.pad(
            cropped_img, 
            padding, 
            mode='constant', 
            value=0.0 # 补零 (黑色)
        )
            
        # 5. 移除 Batch 维度（如果之前添加了）
        return restored_img.squeeze(0) if image.dim() == 3 else restored_img