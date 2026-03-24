import cv2
import numpy as np
from pathlib import Path
from watermarks.core import WatermarkerFactory, BaseWatermarker

# 注册到工厂，该算法不需要特殊的 conda 环境，使用默认环境（None）
@WatermarkerFactory.register("lsb", conda_env=None)
class LSB(BaseWatermarker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 确保 secret 是扁平的 list 或 array
        if isinstance(self.secret, (list, tuple)):
            self.secret = [1, 0, 1, 1, 1, 0, 1]
            self.secret = np.array(self.secret, dtype=int).flatten()
    
    def _func_f(self, x, y):
        """特征函数: LSB(floor(x/2) + y)"""
        return (int(x // 2) + int(y)) % 2

    def embed_one(self, src_path: Path, dst_path: Path):
        # 1. 读取图像
        # LSB 必须使用无损格式读取，避免压缩噪声
        img = cv2.imread(str(src_path))
        if img is None:
            raise FileNotFoundError(f"Image not found: {src_path}")
        
        # 展平图像以便于线性处理像素流
        flat_img = img.flatten()
        
        # 2. 检查容量
        total_pixels = len(flat_img)
        n_pairs = total_pixels // 2
        secret_len = len(self.secret)
        
        if secret_len > n_pairs * 2:
            raise ValueError(f"Secret too long! Image capacity (bits): {n_pairs*2}, Secret length: {secret_len}")
        
        # 3. 嵌入过程 (基于 Mielikainen 2006)
        # 每次取 2 个比特 (m1, m2) 和 2 个像素 (x, y)
        img_encoded = flat_img.copy().astype(np.int16) # 使用 int16 防止 overflow 计算方便
        
        # 补齐 secret 为偶数长度
        bits = self.secret.copy()
        if len(bits) % 2 != 0:
            bits = np.append(bits, 0)
            
        idx_pixel = 0
        for i in range(0, len(bits), 2):
            m1 = bits[i]
            m2 = bits[i+1]
            
            x = img_encoded[idx_pixel]
            y = img_encoded[idx_pixel+1]
            
            lsb_x = x % 2
            func_val = self._func_f(x, y)
            
            if lsb_x == m1:
                # Case 1: LSB(x) 已经匹配 m1
                if func_val != m2:
                    # 需要修改 y，使得 func_f 翻转
                    # 优先向 0-255 区间内调整
                    if y == 255:
                        y -= 1
                    else:
                        y += 1 # 也可以随机 +/- 1，这里简化为 +1
            else:
                # Case 2: LSB(x) 不匹配 m1，必须修改 x
                # 我们希望修改 x 后，func_f(x', y) == m2
                
                # 尝试 x-1
                x_minus = x - 1
                if x > 0 and self._func_f(x_minus, y) == m2:
                    x = x_minus
                # 尝试 x+1
                elif x < 255 and self._func_f(x + 1, y) == m2:
                    x = x + 1
                else:
                    # 如果单纯修改 x 无法满足 func_f 条件
                    # 则 x 必须变 (满足 m1)，y 也必须变 (满足 func_f)
                    if x == 255: x -= 1
                    else: x += 1
                    
                    if y == 255: y -= 1
                    else: y += 1
            
            img_encoded[idx_pixel] = x
            img_encoded[idx_pixel+1] = y
            idx_pixel += 2

        # 4. 保存图像
        # 必须保存为 png 或 bmp 等无损格式，jpg 会破坏 LSB
        img_encoded = img_encoded.astype(np.uint8)
        reshaped_img = img_encoded.reshape(img.shape)
        
        # 强制修改后缀为 .png 防止用户传入 .jpg 导致保存失败或压缩
        save_path = dst_path.with_suffix('.png')
        cv2.imwrite(str(save_path), reshaped_img)

    def extract_one(self, src_path: Path):
        # 1. 读取图像
        img = cv2.imread(str(src_path))
        if img is None:
            raise FileNotFoundError(f"Image not found: {src_path}")
            
        flat_img = img.flatten()
        extracted_bits = []
        
        # 2. 提取过程
        # 根据 secret 的长度提取对应数量的比特
        target_len = len(self.secret)
        # 如果 secret 是奇数，我们在嵌入时补了一位，提取时按偶数对提取后再截断
        read_len = target_len if target_len % 2 == 0 else target_len + 1
        
        idx_pixel = 0
        for _ in range(0, read_len, 2):
            if idx_pixel + 1 >= len(flat_img):
                break
                
            x = flat_img[idx_pixel]
            y = flat_img[idx_pixel+1]
            
            # 恢复 m1
            m1 = x % 2
            # 恢复 m2
            m2 = self._func_f(x, y)
            
            extracted_bits.append(m1)
            extracted_bits.append(m2)
            
            idx_pixel += 2
            
        # 截断到原始 secret 长度
        return extracted_bits[:target_len]
    