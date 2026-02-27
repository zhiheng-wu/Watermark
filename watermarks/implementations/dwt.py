import cv2
import pywt
import numpy as np
from pathlib import Path
from watermarks.core import WatermarkerFactory, BaseWatermarker

@WatermarkerFactory.register("dwt", conda_env=None)
class DWTQIMWatermarker(BaseWatermarker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 算法参数：量化步长 (Step Size)
        # 步长越大，鲁棒性越强，但画质越差；步长越小，画质越好，但易碎。
        self.step = kwargs.get('step', 10.0) 
        self.block_size = 4
        
        # 确保 secret 是扁平 int 数组
        if isinstance(self.secret, (list, tuple)):
            self.secret = np.array(self.secret, dtype=int).flatten()

    def embed_one(self, src_path: Path, dst_path: Path):
        img = cv2.imread(str(src_path))
        if img is None: raise FileNotFoundError(src_path)
        
        # 1. 转换颜色空间 BGR -> YUV
        # 仅在 Y 通道（亮度）嵌入，抗攻击性强
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(img_yuv)
        
        # 2. DWT 变换 (Haar 小波, 1级)
        coeffs = pywt.dwt2(y.astype(float), 'haar')
        LL, (LH, HL, HH) = coeffs
        
        # 选择 HL (水平高频) 子带嵌入
        subband = HL
        h, w = subband.shape
        
        # 3. 分块嵌入
        # 计算容量
        rows = h // self.block_size
        cols = w // self.block_size
        max_bits = rows * cols
        
        secret_bits = self.secret
        if len(secret_bits) > max_bits:
            print(f"[Warning] Secret clipped. Capacity: {max_bits}, Secret: {len(secret_bits)}")
            secret_bits = secret_bits[:max_bits]

        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= len(secret_bits): break
                
                bit = secret_bits[idx]
                
                # 获取 4x4 块
                r_start, r_end = r*self.block_size, (r+1)*self.block_size
                c_start, c_end = c*self.block_size, (c+1)*self.block_size
                block = subband[r_start:r_end, c_start:c_end]
                
                # QIM 核心逻辑：修改块均值
                # 规则：
                # 如果嵌入 1: 量化到 step 的奇数倍 (0.5*step, 1.5*step...)
                # 如果嵌入 0: 量化到 step 的偶数倍 (0.0*step, 1.0*step...)
                mean_val = np.mean(block)
                q = self.step
                
                # 计算当前处于哪个区间
                k = round(mean_val / q)
                
                # 根据 bit 调整 k 为奇数或偶数
                if bit == 1:
                    if k % 2 == 0: k += 1 # 偶变奇
                else: # bit == 0
                    if k % 2 != 0: k += 1 # 奇变偶
                
                target_mean = k * q
                diff = target_mean - mean_val
                
                # 将差值加回 block
                subband[r_start:r_end, c_start:c_end] += diff
                
                idx += 1
        
        # 4. 逆变换与保存
        coeffs_new = (LL, (LH, subband, HH))
        y_watermarked = pywt.idwt2(coeffs_new, 'haar')
        
        # 裁剪越界值并合并
        y_watermarked = np.clip(y_watermarked, 0, 255).astype(np.uint8)
        # 恢复尺寸（DWT可能会导致尺寸微小变化，需resize回原图大小）
        if y_watermarked.shape != y.shape:
             y_watermarked = cv2.resize(y_watermarked, (y.shape[1], y.shape[0]))

        img_merged = cv2.merge([y_watermarked, u, v])
        img_out = cv2.cvtColor(img_merged, cv2.COLOR_YUV2BGR)
        
        cv2.imwrite(str(dst_path), img_out)

    def extract_one(self, src_path: Path):
        img = cv2.imread(str(src_path))
        if img is None: raise FileNotFoundError(src_path)
        
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(img_yuv)
        
        coeffs = pywt.dwt2(y.astype(float), 'haar')
        LL, (LH, HL, HH) = coeffs
        
        subband = HL
        h, w = subband.shape
        rows = h // self.block_size
        cols = w // self.block_size
        
        extracted_bits = []
        target_len = len(self.secret)
        
        for r in range(rows):
            for c in range(cols):
                if len(extracted_bits) >= target_len: break
                
                r_start, r_end = r*self.block_size, (r+1)*self.block_size
                c_start, c_end = c*self.block_size, (c+1)*self.block_size
                block = subband[r_start:r_end, c_start:c_end]
                
                mean_val = np.mean(block)
                q = self.step
                
                # QIM 判决：看最接近奇数倍还是偶数倍
                k = round(mean_val / q)
                if k % 2 != 0:
                    extracted_bits.append(1)
                else:
                    extracted_bits.append(0)
                    
        return extracted_bits