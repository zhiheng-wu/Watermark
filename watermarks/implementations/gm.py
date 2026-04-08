import sys
import os
import json
import random
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

from watermarks.core import WatermarkerFactory, BaseWatermarker

@WatermarkerFactory.register("gm", conda_env="gm")
class GaussMarker(BaseWatermarker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # ================= 1. 动态添加路径 =================
        # 请确保此路径指向您本地 GaussMarker 的源码目录
        GM_ROOT = Path(r"C:/Users/Administrator/Desktop/file/graduation/computer/Watermark/models/GaussMarker")
        if str(GM_ROOT) not in sys.path:
            sys.path.append(str(GM_ROOT))

        try:
            from inverse_stable_diffusion import InversableStableDiffusionPipeline
            from diffusers import DPMSolverMultistepScheduler
            from watermark import Gaussian_Shading_chacha, Gaussian_Shading
            from tr_utils import get_watermarking_pattern, get_watermarking_mask, inject_watermark
            from unet.unet_model import UNet
            from utils import transform_img
        except ImportError as e:
            print(f"[GaussMarker Error] Failed to import modules: {e}")
            raise ImportError("GaussMarker modules not loaded. Please check GM_ROOT path and environment.")

        self.transform_img = transform_img
        self.get_watermarking_pattern = get_watermarking_pattern
        self.get_watermarking_mask = get_watermarking_mask
        self.inject_watermark = inject_watermark

        # === 2. 配置参数 (Args Mock) ===
        class ArgsMock: pass
        self.args = ArgsMock()
        
        # 基础模型与生成参数
        self.args.model_path = kwargs.get('model_path', 'C:/Users/Administrator/Desktop/file/graduation/computer/Watermark/models/stable-diffusion-2-1-base')
        self.args.image_length = kwargs.get('image_length', 512)
        self.args.num_inference_steps = kwargs.get('num_inference_steps', 50)
        self.args.num_inversion_steps = kwargs.get('num_inversion_steps', 50)
        self.args.guidance_scale = kwargs.get('guidance_scale', 7.5)
        
        # GaussMarker 核心参数
        self.args.channel_copy = kwargs.get('channel_copy', 1)
        self.args.w_copy = kwargs.get('w_copy', 8)
        self.args.h_copy = kwargs.get('h_copy', 8)
        self.args.user_number = kwargs.get('user_number', 1000000)
        self.args.fpr = kwargs.get('fpr', 0.000001)
        self.args.chacha = kwargs.get('chacha', True)  # 默认开启流密码增强安全性
        self.args.gen_seed = kwargs.get('seed', 42)
        
        # TreeRing 混合参数 (用于 gt_patch)
        self.args.w_seed = kwargs.get('w_seed', 42)
        self.args.w_channel = kwargs.get('w_channel', 3)
        self.args.w_pattern = kwargs.get('w_pattern', 'ring')
        self.args.w_mask_shape = kwargs.get('w_mask_shape', 'circle')
        self.args.w_radius = kwargs.get('w_radius', 4)
        self.args.w_measurement = kwargs.get('w_measurement', 'l1_complex')
        self.args.w_injection = kwargs.get('w_injection', 'complex')
        self.args.w_pattern_const = kwargs.get('w_pattern_const', 0)

        # 工作目录与密钥路径配置
        self.work_dir = Path(kwargs.get('work_dir', './models/gaussmarker_workspace'))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.args.w1_path = str(self.work_dir / 'w1.pth')
        self.args.w2_path = str(self.work_dir / 'w2.pth')
        
        # GNR 模型配置
        self.gnr_path = kwargs.get('gnr_path', None)
        self.args.classifier_type = kwargs.get('classifier_type', 0)
        self.args.model_nf = kwargs.get('model_nf', 128)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # === 3. 初始化 Pipeline ===
        print(f"[GaussMarker] Initializing InversableStableDiffusionPipeline on {self.device}...")
        scheduler = DPMSolverMultistepScheduler.from_pretrained(self.args.model_path, subfolder='scheduler')
        self.pipe = InversableStableDiffusionPipeline.from_pretrained(
            self.args.model_path,
            scheduler=scheduler,
            torch_dtype=torch.float16,
            revision='fp16',
        ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.safety_checker = None
        
        # 缓存用于反演的空文本 embedding
        self.text_embeddings = self.pipe.get_text_embedding('')

        # === 4. 初始化水印算法与密钥 ===
        self.Gaussian_Shading_chacha = Gaussian_Shading_chacha
        self.Gaussian_Shading = Gaussian_Shading
        self._init_watermark_keys()

        # === 5. 加载 GNR 模型 (仅在提取/评估阶段生效) ===
        self.gnr_model = None
        if self.gnr_path and os.path.exists(self.gnr_path):
            print(f"[GaussMarker] Loading GNR Model from {self.gnr_path}")
            self.gnr_model = UNet(8 if self.args.classifier_type == 1 else 4, 4, nf=self.args.model_nf).to(self.device)
            self.gnr_model.load_state_dict(torch.load(self.gnr_path, map_location=self.device))
            self.gnr_model.eval()
        elif self.gnr_path:
            print(f"[GaussMarker Warning] GNR path provided ({self.gnr_path}) but file not found. Will extract without GNR.")

    def _init_watermark_keys(self):
        """初始化或加载 w1.pth (Gauss密钥) 和 w2.pth (TreeRing Patch)"""
        # 1. 初始化 Gauss 水印 (包含 m, w, key)
        if self.args.chacha:
            if os.path.exists(self.args.w1_path):
                w_info = torch.load(self.args.w1_path, weights_only=False)
                self.watermark = self.Gaussian_Shading_chacha(self.args.channel_copy, self.args.w_copy, self.args.h_copy, self.args.fpr, self.args.user_number, watermark=w_info["w"], m=w_info["m"], key=w_info["key"], nonce=w_info["nonce"])
            else:
                self.watermark = self.Gaussian_Shading_chacha(self.args.channel_copy, self.args.w_copy, self.args.h_copy, self.args.fpr, self.args.user_number)
                _ = self.watermark.create_watermark_and_return_w_m()
                torch.save({"w": self.watermark.watermark, "m": self.watermark.m, "key": self.watermark.key, "nonce": self.watermark.nonce}, self.args.w1_path)
        else:
            if os.path.exists(self.args.w1_path):
                w_info = torch.load(self.args.w1_path)
                self.watermark = self.Gaussian_Shading(self.args.channel_copy, self.args.w_copy, self.args.h_copy, self.args.fpr, self.args.user_number, watermark=w_info["w"], m=w_info["m"], key=w_info["key"])
            else:
                self.watermark = self.Gaussian_Shading(self.args.channel_copy, self.args.w_copy, self.args.h_copy, self.args.fpr, self.args.user_number)
                _ = self.watermark.create_watermark_and_return_w_m()
                torch.save({"w": self.watermark.watermark, "m": self.watermark.m, "key": self.watermark.key}, self.args.w1_path)
        
        self.m_tensor = torch.from_numpy(self.watermark.m).reshape(1, 4, 64, 64).to(self.device)

        # 2. 初始化 TreeRing Patch
        if os.path.exists(self.args.w2_path):
            self.gt_patch = torch.load(self.args.w2_path).to(self.device)
        else:
            self.gt_patch = self.get_watermarking_pattern(self.pipe, self.args, self.device, shape=(1, 4, 64, 64))
            torch.save(self.gt_patch, self.args.w2_path)
            
        self.watermarking_mask = self.get_watermarking_mask(self.gt_patch.real, self.args, self.device)

    # ========================================================
    # 核心接口覆写
    # ========================================================
    def embed_batch(self, input_dir: str, output_dir: str, **kwargs):
        """
        批量生成并嵌入水印。支持解析 .txt, .json 或 HuggingFace Dataset。
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        max_images = kwargs.get('max_images', 1000)
        start_index = kwargs.get('start_index', 0)
        
        prompts = []
        prompt_indices = [] 
        
        # === 提示词解析模块 (完全复用您的优质逻辑) ===
        if input_path.is_file():
            if input_path.suffix == '.json':
                with open(input_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    prompts = [item['prompt'] if isinstance(item, dict) else item for item in data]
            else:
                with open(input_path, 'r', encoding='utf-8') as f:
                    prompts = [line.strip() for line in f if line.strip()]
            prompt_indices = list(range(start_index, start_index + len(prompts)))
            
        else:
            try:
                from datasets import load_from_disk
                print(f"[GaussMarker] 从数据集目录加载: {input_path}")
                dataset = load_from_disk(str(input_path))
                
                all_prompts = dataset['train']['Prompt'] if 'train' in dataset else dataset['Prompt']
                
                total_range = 20000
                actual_range = min(total_range, len(all_prompts))
                if max_images > actual_range:
                    print(f"[GaussMarker] 警告：请求生成数量超过有效范围，将使用 {actual_range}")
                    max_images = actual_range
                
                random.seed(self.seed)
                random_indices = random.sample(range(actual_range), max_images)
                
                prompts = [all_prompts[i] for i in random_indices]
                prompt_indices = random_indices
                print(f"[GaussMarker] 成功抽样 {len(prompts)} 个提示词。")
            except Exception as e:
                raise RuntimeError(f"[GaussMarker Error] 数据集加载失败: {e}")
        
        if not prompts:
            raise ValueError("[GaussMarker] 没有找到可用的提示词")
        # === 批量生成模块 ===
        print(f"[GaussMarker] 嵌入开始，将生成 {len(prompts)} 张图片至 {output_path}")
        
        for idx, (original_index, prompt) in enumerate(zip(prompt_indices, prompts)):
            dst_filename = f"{original_index:06d}.png"
            dst_path = output_path / dst_filename
            
            if dst_path.exists():
                print(f"  [跳过] 文件已存在: {dst_filename}")
                continue
                
            try:
                # 结合基础种子和索引，确保每次生成的随机性可复现
                current_seed = self.args.gen_seed
                self.embed_one_from_prompt(prompt, dst_path, seed=current_seed)
            except Exception as e:
                print(f"  [嵌入失败] 索引 {original_index} (当前进度 {idx}): {e}")
                
        print(f"[GaussMarker] 嵌入阶段生成完毕。")

        # === 架构扩展：生成完毕后，如果配置了需要训练 GNR，则自动触发 ===
        if kwargs.get('train_gnr_after_embed', True):
            self.train_gnr(
                train_steps=kwargs.get('gnr_train_steps', 10000), 
                batch_size=kwargs.get('gnr_batch_size', 16)
            )

    def embed_one(self, src_path: Path, dst_path: Path):
        raise NotImplementedError("GaussMarker is a generative watermark. Call `embed_batch` with a prompt file instead.")

    def embed_one_from_prompt(self, prompt: str, dst_path: Path, seed: int = 42):
        """核心生成逻辑"""
        import utils
        utils.set_random_seed(seed)
        
        # 生成隐向量中的高斯阴影水印
        init_latents_w_gs, _ = self.watermark.create_watermark_and_return_w_m()
        
        # 注入复合水印 (TreeRing Mask + GaussMarker)
        init_latents_w = self.inject_watermark(
            init_latents_w_gs.float().cuda(), 
            self.watermarking_mask, 
            self.gt_patch, 
            self.args
        ).half()

        # SD 推理生成图片
        image_w = self.pipe(
            prompt,
            num_images_per_prompt=1,
            guidance_scale=self.args.guidance_scale,
            num_inference_steps=self.args.num_inference_steps,
            height=self.args.image_length,
            width=self.args.image_length,
            latents=init_latents_w,
        ).images[0]
        
        image_w.save(dst_path)

    def extract_one(self, src_path: Path):
        """
        核心检测逻辑：
        1. 图像转 Tensor。
        2. DDIM Inversion 获取逆向噪声。
        3. 利用 GNR (如已加载) 修复攻击后的掩码。
        4. 解析出水印比特序列。
        """
        img_pil = cv2.imread(str(src_path))
        if img_pil is None: raise FileNotFoundError(src_path)
        img_pil = cv2.cvtColor(img_pil, cv2.COLOR_BGR2RGB)
        
        img_tensor = self.transform_img(img_pil).unsqueeze(0).to(self.text_embeddings.dtype).to(self.device)

        with torch.no_grad():
            image_latents = self.pipe.get_image_latents(img_tensor, sample=False)
            
            # DDIM 逆向去噪
            reversed_latents = self.pipe.forward_diffusion(
                latents=image_latents,
                text_embeddings=self.text_embeddings,
                guidance_scale=1.0, # 逆向时通常设为1
                num_inference_steps=self.args.num_inversion_steps,
            )
            
            reversed_m = (reversed_latents > 0).float()
            
            # 使用 GNR 网络进行修复抗攻击 (鲁棒性核心)
            if self.gnr_model is not None:
                if self.args.classifier_type == 1:
                    input_tensor = torch.cat([self.m_tensor.float(), reversed_m], dim=1)
                else:
                    input_tensor = reversed_m
                restored_m = (F.sigmoid(self.gnr_model(input_tensor)) > 0.5).int()
            else:
                # 若无 GNR 模型，则使用未经修复的掩码
                restored_m = reversed_m.int()
                
            # 从掩码中预测比特位
            pred_w = self.watermark.pred_w_from_m(restored_m)
            
        return pred_w # 返回预测出的 bit 列表

    def calculate_distance(self, original_unused, extracted_w):
        """
        统一距离测算：计算比特误码率 (Bit Error Rate, BER)。
        值越小（越接近0），说明提取出的序列和真实密钥越一致。
        """
        # GaussMarker 的原始密钥储存在 self.watermark.watermark 中
        gt_w = self.watermark.watermark
        
        if len(gt_w) != len(extracted_w): return 1.0
        
        # 计算 Bit Accuracy
        bit_accuracy = (np.array(extracted_w) == np.array(gt_w)).astype(float).mean()
        
        # 转换为 Distance (BER)
        ber = 1.0 - bit_accuracy
        return float(ber)

    def train_gnr(self, train_steps=10000, batch_size=16):
        """
        独立调用的 GNR 训练接口。
        通过包装原项目 train_GNR.py 的逻辑，将其集成在 Factory 流程中。
        """
        print(f"[GaussMarker] 启动 GNR 模型训练 (Steps: {train_steps}, Batch Size: {batch_size})...")
        try:
            import train_GNR
            
            # 将 Factory 层的参数传递给原作者的训练脚本配置
            train_args = self.args
            train_args.train_steps = train_steps
            train_args.batch_size = batch_size
            train_args.lr = 1e-4
            train_args.sample_type = "m"
            train_args.r = 8.0
            train_args.t = 0.0
            train_args.s_min = 0.5
            train_args.s_max = 2.0
            train_args.sh = 0.0
            train_args.fp = 0.0
            train_args.neg_p = 0.5
            train_args.num_workers = 4 # 减小 worker 防止 Windows 下 multiprocessing 报错
            train_args.num_watermarks = 1
            train_args.output_path = str(self.work_dir / 'GNR_Output')
            train_args.w_info_path = self.args.w1_path
            
            # 执行训练
            train_GNR.main(train_args)
            
            print(f"[GaussMarker] GNR 训练完成！模型已保存至 {train_args.output_path}")
            # 更新 gnr_path 以便立刻可用于提取
            self.gnr_path = str(Path(train_args.output_path) / "model_final.pth")
            
        except ImportError:
            print("[GaussMarker Error] 找不到 train_GNR.py，无法执行训练。")
        except Exception as e:
            print(f"[GaussMarker Error] 训练 GNR 过程中发生错误: {e}")