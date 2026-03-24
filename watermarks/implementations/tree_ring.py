import sys
import os
import json
import copy
import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

from watermarks.core import WatermarkerFactory, BaseWatermarker
@WatermarkerFactory.register("tree_ring", conda_env="tree_ring")
class TreeRingWatermarker(BaseWatermarker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # ================= 1. 动态添加路径 =================
        # 指向 tree-ring-watermark 的源码目录，以便能 import 其自定义库
        # tree-ring 源码放在 D:/graduation/computer/Watermark/models/watermarker/tree-ring-watermark
        TR_ROOT = Path(r"D:/graduation/computer/Watermark/models/watermarker/tree-ring-watermark")
        if str(TR_ROOT) not in sys.path:
            sys.path.append(str(TR_ROOT))

        try:
            from inverse_stable_diffusion import InversableStableDiffusionPipeline
            from diffusers import DPMSolverMultistepScheduler
            from optim_utils import get_watermarking_pattern, get_watermarking_mask, inject_watermark, eval_watermark
        except ImportError as e:
            print(f"[TreeRing Error] Failed to import modules: {e}")
            print(f"Please ensure {TR_ROOT} exists and contains 'optim_utils.py', etc.")
            raise ImportError("Tree-Ring modules not loaded.")

        # === 2. 配置参数 (Args Mock) ===
        # 将 kwargs 转化为类似 argparse 的命名空间，因为原代码高度依赖 args 对象
        class ArgsMock:
            pass
        self.args = ArgsMock()
        
        # 模型与生成参数
        self.args.model_id = kwargs.get('model_id', 'D:/graduation/computer/Watermark/models/sd-2.1-base')
        self.args.image_length = kwargs.get('image_length', 512)
        self.args.num_inference_steps = kwargs.get('num_inference_steps', 50)
        self.args.test_num_inference_steps = kwargs.get('test_num_inference_steps', 50)
        self.args.guidance_scale = kwargs.get('guidance_scale', 7.5)
        self.args.num_images = 1
        
        # 水印参数
        self.args.w_seed = kwargs.get('seed', 999999)
        self.args.w_channel = kwargs.get('w_channel', 3) # 原代码中默认有可能是 0 或 3 (取决于你想加在哪里)
        self.args.w_pattern = kwargs.get('w_pattern', 'ring') # 'ring', 'rand', 'zeros'
        self.args.w_mask_shape = kwargs.get('w_mask_shape', 'circle')
        self.args.w_radius = kwargs.get('w_radius', 10)
        self.args.w_measurement = kwargs.get('w_measurement', 'l1_complex')
        self.args.w_injection = kwargs.get('w_injection', 'complex')
        self.args.w_pattern_const = kwargs.get('w_pattern_const', 0)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # === 3. 初始化 Pipeline ===
        print(f"[TreeRing] Initializing InversableStableDiffusionPipeline on {self.device}...")
        scheduler = DPMSolverMultistepScheduler.from_pretrained(self.args.model_id, subfolder='scheduler')
        self.pipe = InversableStableDiffusionPipeline.from_pretrained(
            self.args.model_id,
            scheduler=scheduler,
            torch_dtype=torch.float16,
            revision='fp16',
            safety_checker=None,
            image_processor=None
        ).to(self.device)

        # 缓存用于反演的空文本 embedding
        self.tester_prompt = ''
        self.text_embeddings = self.pipe.get_text_embedding(self.tester_prompt)

        # 生成 Ground Truth 的水印图案 (Key)
        # 注意需要导入原作者的工具函数
        self.get_watermarking_pattern = get_watermarking_pattern
        self.get_watermarking_mask = get_watermarking_mask
        self.inject_watermark = inject_watermark
        self.eval_watermark = eval_watermark
        
        self.gt_patch = self.get_watermarking_pattern(self.pipe, self.args, self.device)
        self.watermarking_mask = None # 在实际生成时确定形状

    # ========================================================
    # 覆写 embed_batch：从包含提示词的文件中读取，而不是读图片
    # ========================================================
    # 修改 embed_batch 方法
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
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 从 kwargs 获取参数
        max_images = kwargs.get('max_images', 1000)
        start_index = kwargs.get('start_index', 0)
        
        prompts = []
        
        if input_path.is_file():
            # 如果是文本文件，按行读取；如果是 json，按列表读取
            if input_path.suffix == '.json':
                with open(input_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    prompts = [item['prompt'] if isinstance(item, dict) else item for item in data]
            else:
                with open(input_path, 'r', encoding='utf-8') as f:
                    prompts = [line.strip() for line in f if line.strip()]
        else:
            # 如果是目录，尝试像之前的代码一样加载数据集
            try:
                from datasets import load_from_disk
                
                print(f"[TreeRing] 从数据集目录加载: {input_path}")
                dataset = load_from_disk(str(input_path))
                
                # 获取提示词，与之前代码保持一致
                if 'train' in dataset:
                    all_prompts = dataset['train']['Prompt']
                else:
                    all_prompts = dataset['Prompt']
                
                # 根据 start_index 和 max_images 截取需要的提示词
                end_idx = min(start_index + max_images, len(all_prompts))
                prompts = all_prompts[start_index:end_idx]
                
                print(f"[TreeRing] 从数据集加载了 {len(prompts)} 个提示词")
                print(f"[TreeRing] 索引范围: {start_index} 到 {end_idx-1}")
                
            except ImportError as e:
                print(f"[TreeRing Error] 需要安装 datasets 库: {e}")
                raise
            except Exception as e:
                print(f"[TreeRing Error] 加载数据集失败: {e}")
                raise
        
        if not prompts:
            raise ValueError("[TreeRing] 没有找到可用的提示词")
        
        print(f"[TreeRing] 嵌入开始，从 {len(prompts)} 个提示词生成图片...")
        print(f"[TreeRing] 输出目录: {output_path}")
        
        for idx, prompt in enumerate(tqdm(prompts)):
            # 生成文件名，使用6位数字编号，与之前的代码保持一致
            file_index = start_index + idx
            dst_filename = f"{file_index:06d}.png"
            dst_path = output_path / dst_filename
            
            # 如果文件已存在，跳过
            if dst_path.exists():
                print(f"  [跳过] 文件已存在: {dst_filename}")
                continue
                
            try:
                # 使用基础种子加上索引确保可重复性
                seed = self.args.w_seed
                self.embed_one_from_prompt(prompt, dst_path, seed=seed)
            except Exception as e:
                print(f"  [嵌入失败] 索引 {file_index} (提示词 {idx}): {e}")
        
        print(f"[TreeRing] 嵌入完成，图片已保存到: {output_path}")

    def embed_one(self, src_path: Path, dst_path: Path):
        """
        保留此接口以满足抽象类，但直接抛出异常提醒调用者。
        年轮水印不需要源图像。
        """
        raise NotImplementedError("Tree-Ring is a generative watermark. Use `embed_one_from_prompt` instead or call `embed_batch` with a prompt file.")

    def embed_one_from_prompt(self, prompt: str, dst_path: Path, seed: int = 42):
        """
        核心生成+嵌入逻辑
        """
        # 设置随机数种子，获取初始 latent (Noise)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        init_latents_w = self.pipe.get_random_latents()
        
        # 获取频域掩码并注入水印 (Key) 到初始噪声中
        if self.watermarking_mask is None:
            self.watermarking_mask = self.get_watermarking_mask(init_latents_w, self.args, self.device)
            
        init_latents_w = self.inject_watermark(init_latents_w, self.watermarking_mask, self.gt_patch, self.args)
        
        # 生成图片
        outputs_w = self.pipe(
            prompt,
            num_images_per_prompt=self.args.num_images,
            guidance_scale=self.args.guidance_scale,
            num_inference_steps=self.args.num_inference_steps,
            height=self.args.image_length,
            width=self.args.image_length,
            latents=init_latents_w,
        )
        orig_image_w = outputs_w.images[0]
        
        # 保存图片
        orig_image_w.save(dst_path)


    def extract_one(self, src_path: Path):
        """
        检测阶段：使用 DDIM Inversion 获取待检测图片的初始噪声，并在频域测算距离。
        注意：提取模式返回值通常是 boolean（是否检测到）或者具体的特征向量。
        为了配合 Factory 中 calculate_distance 的行为，我们这里返回逆向测算的 "距离/指标"。
        """
        img_pil = cv2.imread(str(src_path))
        if img_pil is None: raise FileNotFoundError(src_path)
        img_pil = cv2.cvtColor(img_pil, cv2.COLOR_BGR2RGB)
        
        # 需要将 np 转换回符合 diffusers 预处理要求的 tensor
        from optim_utils import transform_img
        # transform_img 是作者提供的归一化函数
        img_tensor = transform_img(img_pil).unsqueeze(0).to(self.text_embeddings.dtype).to(self.device)

        with torch.no_grad():
            # 1. 编码到潜空间
            image_latents = self.pipe.get_image_latents(img_tensor, sample=False)
            
            # 2. DDIM Inversion 逆向扩散 (去噪)
            reversed_latents = self.pipe.forward_diffusion(
                latents=image_latents,
                text_embeddings=self.text_embeddings,
                guidance_scale=1,
                num_inference_steps=self.args.test_num_inference_steps,
            )
            
            # 3. 评估指标 (测算逆向噪声中是否含有 Key)
            if self.watermarking_mask is None:
                self.watermarking_mask = self.get_watermarking_mask(reversed_latents, self.args, self.device)
            
            # 作者源码 eval_watermark 函数设计为计算无水印和有水印的 distance，我们这里只需对单一对象计算
            # 为复用作者代码，这里采用简化的测算逻辑或调用原函数：
            _, w_metric = self.eval_watermark(reversed_latents, reversed_latents, self.watermarking_mask, self.gt_patch, self.args)

        # 这里返回我们提取到的度量值 (metric)
        # Tree-Ring 论文中，w_metric 通常是距离（L1 complex distance）。
        # 这里将其包装为列表返回，因为你父类的 calculate_distance 会做 flat 比较
        return [w_metric]

    def calculate_distance(self, original, extracted):
        """
        由于 Tree-Ring 返回的直接就是计算好的“距离”（w_metric），
        我们复写 distance 计算：直接返回 extracted 的值。
        """
        # extracted[0] 即是 eval_watermark 算出的 L1 Complex 距离
        return float(extracted[0])