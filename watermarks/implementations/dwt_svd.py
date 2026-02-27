import cv2
import pywt
import numpy as np
from pathlib import Path
from watermarks.core import WatermarkerFactory, BaseWatermarker

@WatermarkerFactory.register("dwt_svd", conda_env=None)
class DWTSVDWatermarker(BaseWatermarker):
    def __init__(self, block_size: int = 4, scale: int = 20, **kwargs):
        """
        :param block_size: SVD 处理的块大小 (通常 4x4 或 8x8)
        :param scale: 量化步长，控制嵌入强度
        """
        super().__init__(**kwargs)
        self.block_size = block_size
        self.scale = scale # 奇异值的量化步长
        if isinstance(self.secret, (list, tuple)):
            self.secret = np.array(self.secret, dtype=int).flatten()

    def _process_block_embed(self, block, bit):
        """对单个块进行 SVD 并嵌入比特"""
        # SVD 分解
        try:
            u, s, vh = np.linalg.svd(block, full_matrices=False)
        except np.linalg.LinAlgError:
            return block # 奇异矩阵忽略

        # 修改最大的奇异值 s[0]
        # 使用 QIM 方法
        sigma = s[0]
        q_idx = round(sigma / self.scale)
        
        if q_idx % 2 != bit:
            if sigma >= q_idx * self.scale:
                q_idx += 1
            else:
                q_idx -= 1
        
        s[0] = q_idx * self.scale
        
        # 重构块
        return np.dot(u * s, vh)

    def _process_block_extract(self, block):
        """从单个块提取比特"""
        try:
            _, s, _ = np.linalg.svd(block, full_matrices=False)
        except np.linalg.LinAlgError:
            return 0
            
        sigma = s[0]
        q_idx = round(sigma / self.scale)
        return q_idx % 2

    def embed_one(self, src_path: Path, dst_path: Path):
        img = cv2.imread(str(src_path))
        if img is None: raise FileNotFoundError(f"{src_path}")

        # 1. 预处理：只取 Y 通道
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        y_channel = img_yuv[:, :, 0].astype(np.float32)
        
        # 2. DWT 变换
        # 使用 'haar' 做一级变换
        coeffs = pywt.dwt2(y_channel, 'haar')
        LL, (LH, HL, HH) = coeffs
        
        # 3. 在 LL 子带上进行分块 SVD 嵌入
        # 这种方法抗压缩能力极强，但对画质影响比在高频嵌入大
        h_ll, w_ll = LL.shape
        
        # 确保能容纳 secret
        num_blocks = (h_ll // self.block_size) * (w_ll // self.block_size)
        if len(self.secret) > num_blocks:
             raise ValueError(f"Secret too long. Capacity: {num_blocks} bits")
        
        bit_idx = 0
        LL_embedded = LL.copy()
        
        for r in range(0, h_ll - self.block_size + 1, self.block_size):
            for c in range(0, w_ll - self.block_size + 1, self.block_size):
                if bit_idx >= len(self.secret):
                    break
                
                block = LL[r:r+self.block_size, c:c+self.block_size]
                bit = self.secret[bit_idx]
                
                LL_embedded[r:r+self.block_size, c:c+self.block_size] = \
                    self._process_block_embed(block, bit)
                
                bit_idx += 1

        # 4. IDWT 重构
        coeffs_embedded = (LL_embedded, (LH, HL, HH))
        y_embedded = pywt.idwt2(coeffs_embedded, 'haar')
        
        # 裁剪尺寸并合并
        h, w = y_channel.shape
        y_embedded = y_embedded[:h, :w]
        
        img_yuv[:, :, 0] = np.clip(y_embedded, 0, 255).astype(np.uint8)
        img_out = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        
        cv2.imwrite(str(dst_path.with_suffix('.png')), img_out)

    def extract_one(self, src_path: Path):
        img = cv2.imread(str(src_path))
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        y_channel = img_yuv[:, :, 0].astype(np.float32)
        
        coeffs = pywt.dwt2(y_channel, 'haar')
        LL, _ = coeffs
        
        h_ll, w_ll = LL.shape
        extracted = []
        
        bit_idx = 0
        target_len = len(self.secret)
        
        for r in range(0, h_ll - self.block_size + 1, self.block_size):
            for c in range(0, w_ll - self.block_size + 1, self.block_size):
                if bit_idx >= target_len:
                    break
                
                block = LL[r:r+self.block_size, c:c+self.block_size]
                extracted.append(self._process_block_extract(block))
                bit_idx += 1
                
        return extracted