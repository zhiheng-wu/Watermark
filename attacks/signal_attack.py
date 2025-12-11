import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import io
import numpy as np
from .core import BaseAttacker, AttackerFactory

# 假设 BaseAttacker 和 AttackerFactory 已经定义 (同上文)

# 1. 高斯噪声攻击 (Gaussian Noise) - 留在 GPU
# --------------------------------------------------
@AttackerFactory.register("gaussian_noise")
class GaussianNoiseAttacker(BaseAttacker):
    """
    高斯噪声攻击。
    - params['std']: 噪声标准差 (float)，默认 0.1。
    - 全程在 GPU 上运行，利用 torch 的并行生成能力。
    """
    def attack(self, image: torch.Tensor) -> torch.Tensor:
        std = self.params.get('std', 0.1)
        
        # 生成与原图形状一致的高斯噪声，直接在 GPU 上生成
        noise = torch.randn_like(image) * std
        
        # 叠加噪声
        adv_image = image + noise
        
        # 截断操作 (Clip)，保证像素值在合法范围内 (假设图像范围 0-1)
        adv_image = torch.clamp(adv_image, 0.0, 1.0)
        
        return adv_image


# 2. 高斯滤波攻击 (Gaussian Blur) - 留在 GPU
# --------------------------------------------------
@AttackerFactory.register("gaussian_blur")
class GaussianBlurAttacker(BaseAttacker):
    """
    高斯模糊攻击。
    - params['kernel_size']: 卷积核大小 (int)，必须是奇数，默认 3。
    - params['sigma']: 高斯核的标准差 (float)。
    - 使用 torchvision 的优化实现，在 GPU 上利用 CUDA 卷积加速。
    """
    def attack(self, image: torch.Tensor) -> torch.Tensor:
        kernel_size = self.params.get('kernel_size', 3)
        sigma = self.params.get('sigma', 1.0)
        
        # 确保 kernel_size 是奇数
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        # TF.gaussian_blur 支持 GPU Tensor，底层调用 CUDA 卷积
        adv_image = TF.gaussian_blur(image, kernel_size=kernel_size, sigma=sigma)
        
        return adv_image


# 3. JPEG 压缩攻击 (JPEG Compression) - CPU 中转
# --------------------------------------------------
@AttackerFactory.register("jpeg_compression")
class JPEGAttacker(BaseAttacker):
    """
    JPEG 压缩攻击。
    - params['quality']: JPEG 质量因子 (int), 1-100，默认 75。
    - 策略: GPU Tensor -> CPU -> PIL (压缩/解压) -> GPU Tensor。
    - 原因: PyTorch 无原生标准 JPEG 编解码算子，CPU 库 (PIL/libjpeg) 更标准且稳定。
    """
    def attack(self, image: torch.Tensor) -> torch.Tensor:
        quality = self.params.get('quality', 75)
        
        # 1. 记录原始设备，以便最后迁回
        original_device = image.device
        
        # 2. 数据准备：去 Batch 维 -> 迁移到 CPU -> 转为 NumPy
        # 假设输入是 (C, H, W) 或 (1, C, H, W)，且范围 0-1
        img_cpu = image.detach().cpu()
        if img_cpu.dim() == 4:
            img_cpu = img_cpu.squeeze(0)
            
        # 转换为 (H, W, C) 格式并量化到 0-255 用于 JPEG 编码
        img_np = (img_cpu.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        # 3. 使用 PIL 进行模拟内存中的 JPEG 读写
        pil_image = Image.fromarray(img_np)
        buffer = io.BytesIO()
        
        # 编码 (压缩)
        pil_image.save(buffer, format="JPEG", quality=quality)
        
        # 解码 (解压，此时产生了压缩伪影)
        buffer.seek(0) # 重置指针
        jpeg_pil = Image.open(buffer)
        
        # 4. 数据恢复：NumPy -> Tensor -> 归一化 -> 迁回 GPU
        # 转换回 (C, H, W)
        jpeg_np = np.array(jpeg_pil)
        
        # 处理灰度图和彩色图的维度差异
        if jpeg_np.ndim == 2:
            # (H, W) -> (H, W, 1)
            jpeg_np = jpeg_np[:, :, np.newaxis]
            
        jpeg_tensor = torch.from_numpy(jpeg_np).permute(2, 0, 1).float() / 255.0
        
        # 如果原图有 Batch 维度，加回去
        if image.dim() == 4:
            jpeg_tensor = jpeg_tensor.unsqueeze(0)
            
        # 5. 迁回原始 GPU
        return jpeg_tensor.to(original_device)