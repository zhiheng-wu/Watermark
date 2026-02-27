import sys
import os
import torch
import cv2
import numpy as np
import json
from pathlib import Path
from torchvision import transforms



from watermarks.core import WatermarkerFactory, BaseWatermarker

@WatermarkerFactory.register("mbrs", conda_env="mbrs")
class MBRSWatermarker(BaseWatermarker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # ================= 动态添加 MBRS 路径 =================
        # 假设 MBRS 文件夹在 models/watermarker/MBRS
        sys.path.append("D:/graduation/computer/Watermark/models/watermarker/MBRS")

        # 尝试导入 MBRS 的 Network 类
        # 注意：根据你的描述，Network.py 在 network 文件夹下
        try:
            from models.watermarker.MBRS.network.Network import Network
        except ImportError:
            # 如果 sys.path 没有生效，尝试相对导入或提示错误
            print(f"[MBRS Error] Cannot import 'network.Network'. Please check path")
            Network = None
        if Network is None:
            raise ImportError("MBRS Network class not found.")

        # === 1. 配置参数 ===
        # 这些参数通常需要与训练时的配置一致
        self.H = kwargs.get('H', 128)          # 裁剪块的高度
        self.W = kwargs.get('W', 128)          # 裁剪块的宽度
        self.message_length = kwargs.get('message_length', 64) # 水印长度
        self.noise_layers = kwargs.get('noise_layers', []) # 推理阶段通常不需要噪声层，或者保持默认
        self.device_str = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(self.device_str)
        self.batch_size = 1 # 推理时 batch 为 1
        self.lr = 0.0001 # 占位，推理不用

        # === 2. 初始化模型 ===
        print(f"[MBRS] Loading Network on {self.device_str}...")
        self.net = Network(
            self.H, self.W, 
            self.message_length, 
            self.noise_layers, 
            self.device, 
            self.batch_size, 
            self.lr,
            with_diffusion=False, 
            only_decoder=False # 即使只做提取，为了加载逻辑通用，通常初始化整个
        )

        # === 3. 加载权重 ===
        # 假设权重路径通过 params 传入，或者硬编码默认路径
        default_weight_path = Path(r"D:/graduation/computer/Watermark/models/watermarker/MBRS/results/MBRS_256_m256/models/EC_42.pth")
        weight_path = kwargs.get('weight_path', str(default_weight_path))
        
        if os.path.exists(weight_path):
            # MBRS 的 Network 类封装了 DataParallel，需要注意 state_dict 的 key
            # Network.py 中有 load_model_ed 方法
            try:
                self.net.load_model_ed(weight_path)
                print(f"[MBRS] Weights loaded from {weight_path}")
            except Exception as e:
                print(f"[MBRS] Failed to load weights using internal method: {e}")
                # 备用加载方案：直接加载到 module
                state_dict = torch.load(weight_path, map_location=self.device)
                self.net.encoder_decoder.module.load_state_dict(state_dict)
        else:
            print(f"[MBRS Warning] Weight file not found at {weight_path}. Using random init.")

        # 设为评估模式
        self.net.encoder_decoder.eval()

        # === 4. 数据预处理 ===
        # MBRS 通常期望输入为 Tensor [B, C, H, W]，范围 [0, 1]
        self.to_tensor = transforms.ToTensor() 

    def _get_center_crop(self, img_np):
        """
        从图像中心裁剪出 HxW 的区域
        返回: (crop_img, (top, bottom, left, right))
        """
        h, w, c = img_np.shape
        if h < self.H or w < self.W:
            # 如果原图比裁剪框还小，进行 resize (这就不是 crop 了，而是 resize)
            img_resized = cv2.resize(img_np, (self.W, self.H))
            return img_resized, None
        
        top = (h - self.H) // 2
        left = (w - self.W) // 2
        bottom = top + self.H
        right = left + self.W
        
        crop = img_np[top:bottom, left:right]
        return crop, (top, bottom, left, right)

    def _place_center_crop(self, full_img, crop, coords):
        """将裁剪块放回原图"""
        if coords is None:
            # 说明之前是 resize 过的，现在 resize 回去
            h, w, _ = full_img.shape
            return cv2.resize(crop, (w, h))
            
        top, bottom, left, right = coords
        out_img = full_img.copy()
        out_img[top:bottom, left:right] = crop
        return out_img

    def embed_one(self, src_path: Path, dst_path: Path):
        # 1. 读取
        img = cv2.imread(str(src_path))
        if img is None: raise FileNotFoundError(f"{src_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 转 RGB
        
        # 2. 准备水印
        # 确保 secret 长度正确，不足补0，多余截断
        secret_bits = np.array(self.secret, dtype=np.float32).flatten()
        if len(secret_bits) < self.message_length:
            secret_bits = np.pad(secret_bits, (0, self.message_length - len(secret_bits)), 'constant')
        else:
            secret_bits = secret_bits[:self.message_length]
            
        msg_tensor = torch.from_numpy(secret_bits).unsqueeze(0).to(self.device) # [1, L]
        
        # 3. 中心裁剪
        crop_np, coords = self._get_center_crop(img)
        
        # 4. 预处理 (HWC -> CHW, [0,255] -> [0,1])
        img_tensor = self.to_tensor(crop_np).unsqueeze(0).to(self.device) # [1, 3, H, W]

        # 5. 网络前向传播 (Encode)
        with torch.no_grad():
            # MBRS forward 返回: encoded_images, noised_images, decoded_messages
            # 我们只需要 encoded_images
            encoded_tensor= self.net.encoder_decoder.module.encoder(img_tensor, msg_tensor)
            
        # 6. 后处理 (Tensor -> Numpy, [0,1] -> [0,255])
        encoded_np = encoded_tensor.squeeze(0).cpu().detach().numpy()
        encoded_np = np.transpose(encoded_np, (1, 2, 0)) # CHW -> HWC
        encoded_np = np.clip(encoded_np * 255.0, 0, 255).astype(np.uint8)

        # 7. 贴回原图
        final_img = self._place_center_crop(img, encoded_np, coords)
        
        # 8. 保存 (必须转回 BGR)
        final_img_bgr = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(dst_path), final_img_bgr)

    def extract_one(self, src_path: Path):
        # 1. 读取
        img = cv2.imread(str(src_path))
        if img is None: raise FileNotFoundError(f"{src_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 2. 中心裁剪
        crop_np, _ = self._get_center_crop(img)
        
        # 3. 预处理
        img_tensor = self.to_tensor(crop_np).unsqueeze(0).to(self.device)
        
        # 4. 网络前向传播 (Decode)
        with torch.no_grad():
            # 这里我们需要直接调用 Decoder
            # Network 类中: self.encoder_decoder 是 DataParallel
            # 内部模型是 EncoderDecoder
            # EncoderDecoder 有 self.decoder
            
            # 访问路径: Network -> DataParallel -> EncoderDecoder -> decoder
            decoder = self.net.encoder_decoder.module.decoder
            
            # MBRS decoder output 通常是 logits (未经过 sigmoid) 或概率
            # 查看 Network.py 的 loss 计算: BCEWithLogitsLoss 用于 discriminator，
            # MSE 用于 message。如果用 MSE，说明输出大概率是 sigmoid 后的或者是 raw float。
            # 通常 HiDDeN 架构最后会过 Sigmoid 或者 Tanh。
            # 如果 Network.py 中 decoded_message_error_rate 用 .gt(0.5)，说明输出在 [0,1] 之间。
            decoded_msg = decoder(img_tensor)
            
        # 5. 转回比特流
        pred_bits = decoded_msg.squeeze(0).cpu().detach().numpy()
        # 大于 0.5 为 1，否则为 0
        extracted_bits = (pred_bits > 0.5).astype(int).tolist()
        return extracted_bits