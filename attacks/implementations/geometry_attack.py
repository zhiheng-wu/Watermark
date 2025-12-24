import os
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from attacks.core import BaseAttacker, AttackerFactory

# 通用辅助函数：检查文件扩展名
def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))

@AttackerFactory.register("scaling")
class ScalingAttacker(BaseAttacker):
    """
    缩放攻击。
    决策：运行在 CPU 上，避免频繁的小数据显存IO拷贝带来的延迟。
    """
    def process(self, input_dir: str, output_dir: str):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        scale = self.params.get('scale', 0.8)
        
        for img_file in input_path.iterdir():
            if not is_image_file(img_file.name):
                continue
            
            # 1. 读取并转换为 Tensor (C, H, W)，范围 [0, 1]
            try:
                # 使用 PIL 读取确保兼容性，convert RGB 防止单通道报错
                pil_img = Image.open(img_file).convert("RGB")
                img_tensor = TF.to_tensor(pil_img) 
                
                # 2. 核心逻辑适配
                C, H, W = img_tensor.shape
                # F.interpolate 需要 Batch 维度 (1, C, H, W)
                img_batch = img_tensor.unsqueeze(0)
                
                # 执行缩放
                resized_img = F.interpolate(
                    img_batch, 
                    scale_factor=scale, 
                    mode='bilinear',
                    align_corners=False
                )
                
                _, _, H_new, W_new = resized_img.shape
                
                # 恢复尺寸
                if scale < 1.0:
                    # 补边
                    pad_h = H - H_new
                    pad_w = W - W_new
                    padding = [
                        pad_w // 2, pad_w - pad_w // 2, 
                        pad_h // 2, pad_h - pad_h // 2
                    ]
                    restored_img = F.pad(resized_img, padding, mode='constant', value=0.0)
                    
                elif scale > 1.0:
                    # 裁剪
                    start_h = (H_new - H) // 2
                    start_w = (W_new - W) // 2
                    restored_img = resized_img[:, :, start_h:start_h+H, start_w:start_w+W]
                else:
                    restored_img = resized_img

                # 3. 移除 Batch 维度并保存
                final_tensor = restored_img.squeeze(0)
                
                # 转换回 PIL 并保存
                final_img = TF.to_pil_image(final_tensor)
                final_img.save(output_path / img_file.name, format='PNG')
                
            except Exception as e:
                print(f"[Scaling] Error processing {img_file.name}: {e}")

@AttackerFactory.register("rotation")
class RotationAttacker(BaseAttacker):
    """
    旋转攻击。运行在 CPU 上。
    """
    def process(self, input_dir: str, output_dir: str):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        angle = self.params.get('angle', 5.0)

        for img_file in input_path.iterdir():
            if not is_image_file(img_file.name):
                continue
                
            try:
                pil_img = Image.open(img_file).convert("RGB")
                img_tensor = TF.to_tensor(pil_img) # (C, H, W)
                
                # TF.rotate 可以直接处理 (C, H, W)
                rotated_tensor = TF.rotate(
                    img_tensor, 
                    angle, 
                    fill=0, 
                    interpolation=TF.InterpolationMode.BILINEAR
                )
                
                final_img = TF.to_pil_image(rotated_tensor)
                final_img.save(output_path / img_file.name, format='PNG')
                
            except Exception as e:
                print(f"[Rotation] Error processing {img_file.name}: {e}")

@AttackerFactory.register("cropping")
class CroppingAttacker(BaseAttacker):
    """
    裁剪攻击。运行在 CPU 上。
    """
    def process(self, input_dir: str, output_dir: str):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        crop_ratio = self.params.get('crop_ratio', 0.9)
        crop_start_h = self.params.get('crop_start_h', None)
        crop_start_w = self.params.get('crop_start_w', None)

        for img_file in input_path.iterdir():
            if not is_image_file(img_file.name):
                continue
            
            try:
                pil_img = Image.open(img_file).convert("RGB")
                img_tensor = TF.to_tensor(pil_img) # (C, H, W)
                C, H, W = img_tensor.shape
                
                # 计算参数
                H_crop, W_crop = int(H * crop_ratio), int(W * crop_ratio)
                
                start_h = (H - H_crop) // 2 if crop_start_h is None else max(0, min(crop_start_h, H - H_crop))
                start_w = (W - W_crop) // 2 if crop_start_w is None else max(0, min(crop_start_w, W - W_crop))
                
                # 执行裁剪
                cropped_tensor = img_tensor[:, start_h : start_h + H_crop, start_w : start_w + W_crop]
                
                # 补边恢复原尺寸
                pad_h = H - H_crop
                pad_w = W - W_crop
                padding = [
                    pad_w // 2, pad_w - pad_w // 2,
                    pad_h // 2, pad_h - pad_h // 2
                ]
                
                restored_tensor = F.pad(cropped_tensor, padding, mode='constant', value=0.0)
                
                final_img = TF.to_pil_image(restored_tensor)
                final_img.save(output_path / img_file.name, format='PNG')
                
            except Exception as e:
                print(f"[Cropping] Error processing {img_file.name}: {e}")