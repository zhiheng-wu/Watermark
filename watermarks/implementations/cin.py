import sys
import os
import torch
import numpy as np
import yaml
import cv2
from pathlib import Path
from torchvision import transforms



from watermarks.core import WatermarkerFactory, BaseWatermarker

@WatermarkerFactory.register("cin", conda_env="C:/ProgramData/anaconda3/envs/cin")
class CINWatermarker(BaseWatermarker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # ================= 1. 动态添加路径 =================
        # 指向 CIN 的 codes 目录，以便能 import utils, models 等
        CIN_ROOT = Path(r"D:/graduation/computer/Watermark/models/watermarker/CIN/codes")
        if str(CIN_ROOT) not in sys.path:
            sys.path.append(str(CIN_ROOT))

        # 尝试导入 CIN 的依赖
        try:
            from utils.yml import parse_yml, dict_to_nonedict
            from models.Network import Network
        except ImportError as e:
            print(f"[CIN Error] Failed to import CIN modules: {e}")
            print(f"Please ensure {CIN_ROOT} exists and contains 'models' and 'utils'.")
            Network = None
        if Network is None:
            raise ImportError("CIN modules not loaded.")

        # === 2. 配置参数 (Mock opt) ===
        # 读取原始 YAML 获取网络结构参数
        yml_path = CIN_ROOT / 'options/opt.yml' # 默认配置路径
        if not yml_path.exists():
            raise FileNotFoundError(f"Config not found: {yml_path}")
            
        raw_opt = parse_yml(str(yml_path))
        self.opt = dict_to_nonedict(raw_opt)

        # 强制覆盖为测试模式，防止加载优化器等训练专用组件
        self.opt['train/test'] = 'test'
        self.opt['train']['os_environ'] = '0' # 占位
        
        # 允许通过 kwargs 覆盖 YAML 中的参数
        self.device_str = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(self.device_str)
        
        # 裁剪尺寸
        self.crop_size = kwargs.get('crop_size', 128) # 默认 128，需与权重匹配
        self.message_length = self.opt['network']['message_length'] # 从 yaml 读取

        # === 3. 伪造 path_in 字典 ===
        # Network.__init__ 需要这些路径来创建日志，我们给它指向临时目录避免污染
        temp_root = CIN_ROOT.parent / 'temp_logs'
        path_in = {
            'log_folder': str(temp_root / 'log'),
            'img_w_folder_tra': str(temp_root / 'img_tra'),
            'img_w_folder_val': str(temp_root / 'img_val'),
            'img_w_folder_test': str(temp_root / 'img_test'),
            'loss_w_folder': str(temp_root / 'loss'),
            'path_checkpoint': str(temp_root / 'ckpt'),
            'opt_folder': str(temp_root / 'opt'),
            'time_now_NewExperiment': 'inference_mode'
        }
        # 创建必要的目录防止报错
        os.makedirs(path_in['log_folder'], exist_ok=True)

        # === 4. 初始化网络 ===
        print(f"[CIN] Initializing Network on {self.device_str}...")
        self.model_wrapper = Network(self.opt, self.device, path_in)
        
        # === 5. 加载权重 ===
        # 优先使用 kwargs 传入的 weight_path，否则尝试使用 yaml 中的路径
        weight_path = kwargs.get('weight_path', r"D:/graduation/computer/Watermark/models/watermarker/CIN/pth/cinNet&nsmNet.pth")
        
        if os.path.exists(weight_path):
            print(f"[CIN] Loading weights from {weight_path}")
            try:
                # CIN 的权重格式比较特殊，使用 torch.load 加载字典
                checkpoint = torch.load(weight_path, map_location=self.device)
                
                # 网络实例在 Network.py 中叫 cinNet，而不是 netG
                if hasattr(self.model_wrapper, 'cinNet'):
                    # Network.py 默认套了 DataParallel，所以加 .module
                    if isinstance(self.model_wrapper.cinNet, torch.nn.DataParallel):
                        # strict=False 防止因为一些多余的 loss 参数导致报错
                        self.model_wrapper.cinNet.module.load_state_dict(checkpoint, strict=False)
                    else:
                        self.model_wrapper.cinNet.load_state_dict(checkpoint, strict=False)
                else:
                    print("[CIN Warning] self.cinNet not found in Network class.")
            except Exception as e:
                print(f"[CIN Error] Error loading weights: {e}")
        else:
            print(f"[CIN Warning] Weight file not found at {weight_path}")

        self.model_wrapper.cinNet.eval()
        self.to_tensor = transforms.ToTensor()

    def _get_center_crop(self, img_np):
        """通用中心裁剪逻辑"""
        h, w, c = img_np.shape
        if h < self.crop_size or w < self.crop_size:
            return cv2.resize(img_np, (self.crop_size, self.crop_size)), None
        
        top = (h - self.crop_size) // 2
        left = (w - self.crop_size) // 2
        crop = img_np[top:top+self.crop_size, left:left+self.crop_size]
        return crop, (top, top+self.crop_size, left, left+self.crop_size)

    def _place_center_crop(self, full_img, crop, coords):
        if coords is None:
            h, w, _ = full_img.shape
            return cv2.resize(crop, (w, h))
        t, b, l, r = coords
        out = full_img.copy()
        out[t:b, l:r] = crop
        return out

    def embed_one(self, src_path: Path, dst_path: Path):
        img = cv2.imread(str(src_path))
        if img is None: raise FileNotFoundError(src_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 1. 裁剪
        crop_np, coords = self._get_center_crop(img)
        
        # 2. 准备数据
        # CIN 输入通常是 [Batch, C, H, W]
        img_tensor = self.to_tensor(crop_np).unsqueeze(0).to(self.device)
        
        # 准备水印消息
        # 截断或补零
        secret_bits = np.array(self.secret, dtype=np.float32).flatten()
        if len(secret_bits) < self.message_length:
            secret_bits = np.pad(secret_bits, (0, self.message_length - len(secret_bits)), 'constant')
        else:
            secret_bits = secret_bits[:self.message_length]
        
        msg_tensor = torch.from_numpy(secret_bits).unsqueeze(0).to(self.device)

        # 3. 前向传播 (Embed)
        # 3. 前向传播 (Embed)
        with torch.no_grad():
            # 严格按照 CIN.py 的参数传入: (image, message, noise_choice, is_train)
            # 嵌入时无需施加噪声，传入 'Identity' 或 None (根据你 opt.yml 的配置)
            outputs = self.model_wrapper.cinNet(img_tensor, msg_tensor, 'Identity', False)
            
            # CIN 的输出是 6 个变量，第一个是 watermarking_img
            stego_img = outputs[0]

        # 4. 后处理
        stego_np = stego_img.squeeze(0).cpu().detach().numpy()
        stego_np = np.transpose(stego_np, (1, 2, 0))
        stego_np = np.clip(stego_np * 255.0, 0, 255).astype(np.uint8)

        # 5. 拼回
        final_img = self._place_center_crop(img, stego_np, coords)
        
        # 6. 保存
        cv2.imwrite(str(dst_path), cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))

    def extract_one(self, src_path: Path):
        img = cv2.imread(str(src_path))
        if img is None: raise FileNotFoundError(src_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 1. 裁剪
        crop_np, _ = self._get_center_crop(img)
        img_tensor = self.to_tensor(crop_np).unsqueeze(0).to(self.device)

        # 2. 逆向传播 (Extract)
        # 2. 逆向传播 (Extract)
        with torch.no_grad():
            # CIN 没有 rev=True 的用法。
            # 提取时，把包含水印的图片当做输入，并随便给一个假的 message 占位
            dummy_msg = torch.zeros((1, self.message_length)).to(self.device)
            
            # 再次前向传播
            outputs = self.model_wrapper.cinNet(img_tensor, dummy_msg, 'Identity', False)
            
            # 根据 Network.py，输出的第 6 个元素是 msg_nsm (提取出的消息)
            rec_msg_tensor = outputs[5]

        # 3. 转列表
        pred_bits = rec_msg_tensor.squeeze(0).cpu().detach().numpy()
        # 将输出通过 Sigmoid 激活或直接使用 0 阈值判断 (视原模型损失函数而定，通常 >0 就是 1)
        extracted_bits = (pred_bits > 0).astype(int).tolist()
        
        return extracted_bits