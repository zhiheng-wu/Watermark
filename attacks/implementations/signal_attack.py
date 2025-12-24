import torch
import torchvision.transforms.functional as TF
from PIL import Image
import io
import os
from pathlib import Path
from attacks.core import BaseAttacker, AttackerFactory

# 辅助函数：过滤图片
def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))

# 1. 高斯噪声攻击 (Gaussian Noise) - CPU
# --------------------------------------------------
@AttackerFactory.register("gaussian_noise")
class GaussianNoiseAttacker(BaseAttacker):
    """
    高斯噪声攻击。
    策略: CPU Tensor 操作。
    """
    def process(self, input_dir: str, output_dir: str):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        std = self.params.get('std', 0.1)
        
        for img_file in input_path.iterdir():
            if not is_image_file(img_file.name):
                continue
            
            try:
                # 1. 读取并转为 Tensor (C, H, W), range [0, 1]
                pil_img = Image.open(img_file).convert("RGB")
                img_tensor = TF.to_tensor(pil_img)
                
                # 2. 生成噪声 (CPU)
                noise = torch.randn_like(img_tensor) * std
                
                # 3. 叠加并截断
                adv_image = torch.clamp(img_tensor + noise, 0.0, 1.0)
                
                # 4. 保存
                final_img = TF.to_pil_image(adv_image)
                final_img.save(output_path / img_file.name, format='PNG')
                
            except Exception as e:
                print(f"[GaussianNoise] Error processing {img_file.name}: {e}")


# 2. 高斯滤波攻击 (Gaussian Blur) - CPU
# --------------------------------------------------
@AttackerFactory.register("gaussian_blur")
class GaussianBlurAttacker(BaseAttacker):
    """
    高斯模糊攻击。
    策略: 使用 torchvision 的 CPU 实现。
    """
    def process(self, input_dir: str, output_dir: str):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        kernel_size = self.params.get('kernel_size', 3)
        sigma = self.params.get('sigma', 1.0)
        
        # 确保 kernel_size 是奇数
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        for img_file in input_path.iterdir():
            if not is_image_file(img_file.name):
                continue
            
            try:
                pil_img = Image.open(img_file).convert("RGB")
                img_tensor = TF.to_tensor(pil_img)
                
                # TF.gaussian_blur 在 CPU 上也有很好的优化
                adv_image = TF.gaussian_blur(img_tensor, kernel_size=kernel_size, sigma=sigma)
                
                final_img = TF.to_pil_image(adv_image)
                final_img.save(output_path / img_file.name, format='PNG')
                
            except Exception as e:
                print(f"[GaussianBlur] Error processing {img_file.name}: {e}")


# 3. JPEG 压缩攻击 (JPEG Compression) - CPU (纯内存操作)
# --------------------------------------------------
@AttackerFactory.register("jpeg_compression")
class JPEGAttacker(BaseAttacker):
    """
    JPEG 压缩攻击。
    策略: 文件 -> PIL -> 内存Buffer(JPEG编码) -> PIL -> 文件(PNG保存)。
    注意: 即使中间经过了 JPEG 压缩，最终保存通常建议存为 PNG，
          以无损地保留"JPEG压缩带来的伪影"，防止二次压缩。
    """
    def process(self, input_dir: str, output_dir: str):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        quality = self.params.get('quality', 75)
        
        for img_file in input_path.iterdir():
            if not is_image_file(img_file.name):
                continue
                
            try:
                # 1. 打开原始图片
                img = Image.open(img_file).convert("RGB")
                
                # 2. 创建内存中的二进制流
                buffer = io.BytesIO()
                
                # 3. 将图片以 JPEG 格式保存到内存流中 (这里引入了压缩损伤)
                img.save(buffer, format="JPEG", quality=quality)
                
                # 4. 重新从内存流中读取图片
                buffer.seek(0)
                jpeg_artifact_img = Image.open(buffer)
                
                # 5. 保存结果
                # 注意：这里我们保存为 PNG 格式，是为了"固化" JPEG 造成的像素改变。
                # 如果这里存为 JPEG，可能会发生二次压缩。
                jpeg_artifact_img.save(output_path / img_file.name, format='PNG')
                
            except Exception as e:
                print(f"[JPEG] Error processing {img_file.name}: {e}")