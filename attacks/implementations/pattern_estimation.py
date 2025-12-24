import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from attacks.core import BaseAttacker, AttackerFactory

@AttackerFactory.register("pattern_estimation")
class PatternEstimationAttacker(BaseAttacker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 获取参数
        self.clean_path = self.params.get('clean_path')
        self.watermarked_path = self.params.get('watermarked_path')
        self.num_images = int(self.params.get('num_images', 100))
        # 默认估算尺寸为 512x512，如果图片尺寸不一致，后续会插值调整
        self.init_size = self.params.get('image_size', (512, 512)) 
        self.factor = float(self.params.get('factor', 1.0))
        
        if not self.clean_path or not self.watermarked_path:
            raise ValueError("PatternEstimationAttacker Requires 'clean_path' and 'watermarked_path'.")

        # 【初始化阶段】预计算水印模式 (C, H, W) Tensor，范围理论上在 [-1, 1] 之间
        # 这一步只会在工厂实例化时运行一次
        self.pattern_tensor = self._compute_difference_pattern()

    def _sum_images(self, path, count):
        """
        读取并计算文件夹中前 count 张图片的平均值。
        返回: (C, H, W) Tensor, 范围 [0, 1]
        """
        image_sum = None
        processed_count = 0
        
        # 确保路径存在
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")

        files = sorted([f for f in os.listdir(path) if f.lower().endswith(('png', 'jpg', 'jpeg'))])[:count]

        if not files:
            raise ValueError(f"No images found in {path}")

        print(f"[Pattern] Calculating average for {path} ({len(files)} images)...")
        
        for file in tqdm(files, desc=f"Loading {os.path.basename(path)}"):
            img_path = os.path.join(path, file)
            try:
                # 1. 读取并 Resize 到统一尺寸
                # 使用 convert('RGB') 确保通道一致
                image = Image.open(img_path).resize(self.init_size).convert('RGB')
                
                # 2. 转为 Tensor (C, H, W), 范围 [0, 1]
                img_tensor = TF.to_tensor(image)

                if image_sum is None:
                    image_sum = img_tensor
                else:
                    image_sum += img_tensor
                processed_count += 1
            except Exception as e:
                print(f"Skipping {file}: {e}")
            
        return image_sum / processed_count if processed_count > 0 else None

    def _compute_difference_pattern(self):
        """计算 (Watermarked_Avg - Clean_Avg)"""
        clean_avg = self._sum_images(self.clean_path, self.num_images)
        wm_avg = self._sum_images(self.watermarked_path, self.num_images)
        
        if clean_avg is None or wm_avg is None:
            raise RuntimeError("Failed to compute average images. Check your paths.")

        # 结果形状: (C, H, W)
        return wm_avg - clean_avg

    def process(self, input_dir: str, output_dir: str):
        """
        新的标准接口实现
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 模式 tensor 已经在 __init__ 中计算完毕，驻留在内存中 (CPU)
        pattern = self.pattern_tensor
        
        print(f"[Pattern] Applying estimated pattern removal...")

        for img_file in input_path.iterdir():
            if img_file.suffix.lower() not in ['.png', '.jpg', '.jpeg', '.bmp']:
                continue
            
            try:
                # 1. 读取图片
                pil_img = Image.open(img_file).convert("RGB")
                img_tensor = TF.to_tensor(pil_img) # (C, H, W), [0, 1]
                
                C, H, W = img_tensor.shape
                
                # 2. 动态调整 Pattern 尺寸以匹配当前输入图片
                # self.pattern_tensor 是 (C, Hp, Wp)
                # F.interpolate 需要 (N, C, H, W)
                current_pattern = pattern
                if (pattern.shape[1], pattern.shape[2]) != (H, W):
                    current_pattern = F.interpolate(
                        pattern.unsqueeze(0), 
                        size=(H, W), 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze(0)
                
                # 3. 核心攻击：原图 - 因子 * 噪声模式
                # 确保在同一 device (这里都是 CPU)
                adv_image = img_tensor - (self.factor * current_pattern)
                
                # 4. 截断并保存
                adv_image = torch.clamp(adv_image, 0.0, 1.0)
                
                final_img = TF.to_pil_image(adv_image)
                final_img.save(output_path / img_file.name, format='PNG')
                
            except Exception as e:
                print(f"[Pattern] Error processing {img_file.name}: {e}")