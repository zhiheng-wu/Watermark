import sys
import os
import json
import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

from watermarks.core import WatermarkerFactory, BaseWatermarker

@WatermarkerFactory.register("gs", conda_env="gs")
class GaussianShadingWatermarker(BaseWatermarker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # ================= 1. 动态添加路径 =================
        # 指向 Gaussian-Shading 的源码目录
        GS_ROOT = Path(r"C:/Users/Administrator/Desktop/file/graduation/computer/Watermark/models/Gaussian-Shading")
        if str(GS_ROOT) not in sys.path:
            sys.path.append(str(GS_ROOT))

        try:
            from inverse_stable_diffusion import InversableStableDiffusionPipeline
            from diffusers import DPMSolverMultistepScheduler
            from watermark import Gaussian_Shading, Gaussian_Shading_chacha
            from image_utils import transform_img
        except ImportError as e:
            print(f"[GaussianShading Error] Failed to import modules: {e}")
            raise ImportError("Gaussian-Shading modules not loaded.")

        # === 2. 配置参数 (Args Mock) ===
        class ArgsMock:
            pass
        self.args = ArgsMock()
        
        # 模型路径
        self.args.model_path = kwargs.get('model_path', 'C:/Users/Administrator/Desktop/file/graduation/computer/Watermark/models/stable-diffusion-2-1-base')
        self.args.image_length = kwargs.get('image_length', 512)
        self.args.guidance_scale = kwargs.get('guidance_scale', 7.5)
        self.args.num_inference_steps = kwargs.get('num_inference_steps', 50)
        self.args.num_inversion_steps = kwargs.get('num_inversion_steps', 50)
        
        # 水印核心参数
        self.args.channel_copy = kwargs.get('channel_copy', 1)
        self.args.hw_copy = kwargs.get('hw_copy', 8)
        self.args.fpr = kwargs.get('fpr', 0.000001)
        self.args.user_number = kwargs.get('user_number', 1000000)
        self.args.chacha = kwargs.get('chacha', False)
        self.args.gen_seed = kwargs.get('seed', 42)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # === 3. 初始化 Pipeline ===
        print(f"[GaussianShading] Initializing Pipeline on {self.device}...")
        scheduler = DPMSolverMultistepScheduler.from_pretrained(self.args.model_path, subfolder='scheduler')
        self.pipe = InversableStableDiffusionPipeline.from_pretrained(
            self.args.model_path,
            scheduler=scheduler,
            torch_dtype=torch.float16,
            revision='fp16',
        ).to(self.device)
        self.pipe.safety_checker = None

        # 缓存用于反演的空文本 embedding
        self.tester_prompt = ''
        self.text_embeddings = self.pipe.get_text_embedding(self.tester_prompt)

        # === 4. 初始化水印类 ===
        if self.args.chacha:
            self.watermark = Gaussian_Shading_chacha(
                self.args.channel_copy, self.args.hw_copy, self.args.fpr, self.args.user_number
            )
        else:
            self.watermark = Gaussian_Shading(
                self.args.channel_copy, self.args.hw_copy, self.args.fpr, self.args.user_number
            )
        
        self.transform_img = transform_img

    def embed_batch(self, input_dir: str, output_dir: str, **kwargs):
        """
        覆盖父类方法。
        由于是生成式水印，input_dir 此时应该指向一个包含数据集的目录或一个提示词文件。
        
        Args:
            input_dir: 数据集目录路径或提示词文件路径
            output_dir: 图片输出目录
            **kwargs: 额外参数，包括：
                - max_images: 最大生成图片数量
                - start_index: 起始索引
        """
        import warnings
        warnings.filterwarnings("ignore")  # 屏蔽底层库抛出的无害警告
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 从 kwargs 获取参数
        max_images = kwargs.get('max_images', 1000)
        start_index = kwargs.get('start_index', 0)
        
        prompts = []
        prompt_indices = [] 
        
        if input_path.is_file():
            # 如果是文本文件，按行读取；如果是 json，按列表读取
            if input_path.suffix == '.json':
                with open(input_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    prompts = [item['prompt'] if isinstance(item, dict) else item for item in data]
            else:
                with open(input_path, 'r', encoding='utf-8') as f:
                    prompts = [line.strip() for line in f if line.strip()]
            
            # 文件情况下，使用连续索引
            prompt_indices = list(range(start_index, start_index + len(prompts)))
            
            # 截断以符合 max_images 限制
            prompts = prompts[:max_images]
            prompt_indices = prompt_indices[:max_images]
            
        else:
            try:
                from datasets import load_from_disk
                
                print(f"[GaussianShading] 从数据集目录加载: {input_path}")
                dataset = load_from_disk(str(input_path))
                
                if 'train' in dataset:
                    all_prompts = dataset['train']['Prompt']
                else:
                    all_prompts = dataset['Prompt']
                
                import random
                
                # 从0-20000中随机抽样max_images个索引
                # 首先确定抽样范围
                total_range = 20000
                actual_range = min(total_range, len(all_prompts))
                
                if max_images > actual_range:
                    print(f"[GaussianShading] 警告：最大图片数量 {max_images} 超过有效范围 {actual_range}，将使用 {actual_range}")
                    max_images = actual_range
                
                # 设置随机数种子保证抽样一致性 (可选)
                random.seed(self.args.gen_seed)
                # 随机抽样索引
                random_indices = random.sample(range(actual_range), max_images)
                
                # 根据随机索引获取提示词
                prompts = [all_prompts[i] for i in random_indices]
                prompt_indices = random_indices  # 保存原始索引
                
                print(f"[GaussianShading] 从数据集前 {actual_range} 个提示词中随机抽样了 {len(prompts)} 个提示词")
                print(f"[GaussianShading] 抽样索引示例: {sorted(random_indices)[:10]}...")  # 显示前10个索引
                
            except ImportError as e:
                print(f"[GaussianShading Error] 需要安装 datasets 库: {e}")
                raise
            except Exception as e:
                print(f"[GaussianShading Error] 加载数据集失败: {e}")
                raise
        
        if not prompts:
            raise ValueError("[GaussianShading] 没有找到可用的提示词")
        
        print(f"[GaussianShading] 嵌入开始，从 {len(prompts)} 个提示词生成图片...")
        print(f"[GaussianShading] 输出目录: {output_path}")
        
        for idx, (original_index, prompt) in enumerate(zip(prompt_indices, prompts)):
            # 使用原始数据集中的索引作为文件名
            file_index = original_index
            dst_filename = f"{file_index:06d}.png"
            dst_path = output_path / dst_filename
            
            # 如果文件已存在，跳过
            if dst_path.exists():
                print(f"  [跳过] 文件已存在: {dst_filename}")
                continue
                
            try:
                # 核心：设定种子并调用 GaussianShading 的方法生成潜变量
                # 基础种子加上索引，确保生成过程与作者原代码一致且可重复
                current_seed = self.args.gen_seed
                torch.manual_seed(current_seed) 
                
                init_latents_w = self.watermark.create_watermark_and_return_w()
                
                outputs = self.pipe(
                    prompt,
                    num_images_per_prompt=1,
                    guidance_scale=self.args.guidance_scale,
                    num_inference_steps=self.args.num_inference_steps,
                    height=self.args.image_length,
                    width=self.args.image_length,
                    latents=init_latents_w,
                )
                outputs.images[0].save(dst_path)
            except Exception as e:
                print(f"  [嵌入失败] 原始索引 {file_index} (当前 {idx}): {e}")
        
        print(f"[GaussianShading] 嵌入完成，图片已保存到: {output_path}")

    def embed_one(self, src_path: Path, dst_path: Path):
        raise NotImplementedError("Gaussian-Shading is generative. Use embed_batch with prompts.")

    # ========================================================
    # 检测阶段：反演并计算准确率
    # ========================================================
    def extract_one(self, src_path: Path):
        """
        提取逻辑：将图片逆向扩散回噪声空间，并使用 eval_watermark 计算匹配度
        """
        img_bgr = cv2.imread(str(src_path))
        if img_bgr is None: raise FileNotFoundError(src_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 1. 预处理图片并编码至潜空间
        img_tensor = self.transform_img(img_rgb).unsqueeze(0).to(self.text_embeddings.dtype).to(self.device)
        
        with torch.no_grad():
            image_latents = self.pipe.get_image_latents(img_tensor, sample=False)
            
            # 2. DDIM Inversion 还原初始噪声
            reversed_latents = self.pipe.forward_diffusion(
                latents=image_latents,
                text_embeddings=self.text_embeddings,
                guidance_scale=1,
                num_inference_steps=self.args.num_inversion_steps,
            )

            # 3. 核心评估：计算水印匹配度
            # GaussianShading 的 eval_watermark 通常返回一个 metric (如匹配的 bit 数或相似度)
            acc_metric = self.watermark.eval_watermark(reversed_latents)

        # 返回 [metric] 供 calculate_distance 使用
        return [acc_metric]

    def calculate_distance(self, original, extracted):
        """
        Gaussian-Shading 的 eval_watermark 实际上算的是一种 'Score' (越大越像有水印)。
        为了适配工厂的 is_detected = distance < threshold 逻辑，
        我们需要将 score 转换为 distance，或者直接返回负值，
        但最简单的方法是直接返回 1 - score (假设 score 在 0-1 之间) 或取负。
        
        注意：在 Gaussian Shading 原文中，检测通常基于阈值判定，
        若原代码 eval_watermark 直接返回准确率，此处直接返回该值的反向指标。
        """
        score = float(extracted[0])
        # 假设我们希望 threshold 是一个类似 0.5 的判定线
        # 我们返回 (1.0 - score)，这样 score 越高（越像水印），distance 越小
        return 1.0 - score