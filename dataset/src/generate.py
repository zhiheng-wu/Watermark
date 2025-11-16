import json
import os
import torch
import gc
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers import StableDiffusion3Pipeline
from datasets import load_from_disk
from PIL import Image
import logging
from typing import Dict, List, Optional

class SD3BatchImageGenerator:
    def __init__(self, 
                 model_dir: str,
                 dataset_path: str,
                 output_dir: str = "../origin",
                 checkpoint_file: str = "../jsons/ori_checkpoint.json",
                 max_images: int = 50000,
                 start_index: int = 0):
        """
        SD3批量图片生成器
        
        Args:
            model_dir: 本地模型路径
            dataset_path: 数据集路径
            output_dir: 图片输出目录
            checkpoint_file: 断点续传记录文件
            max_images: 最大生成图片数量
            start_index: 起始索引
        """
        self.model_dir = model_dir
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.checkpoint_file = checkpoint_file
        self.max_images = max_images
        self.start_index = start_index
        self.current_index = start_index
        
        # 初始化设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
        # 加载进度记录
        self.checkpoint_data = self._load_checkpoint()
        
        # 初始化模型（延迟加载）
        self.pipeline = None
        self.generator = None
        
    def _setup_logging(self):
        """设置日志记录"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.output_dir, "generation.log")),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_checkpoint(self) -> Dict:
        """加载或创建检查点文件"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                self.logger.info(f"找到检查点文件，最后处理的索引: {checkpoint.get('last_index', 0)}")
                return checkpoint
            except Exception as e:
                self.logger.error(f"加载检查点失败: {e}，将创建新文件")
        
        # 创建新的检查点
        checkpoint = {
            "last_index": self.start_index - 1,
            "generated_indices": [],
            "failed_indices": [],
            "start_time": None,
            "total_generated": 0
        }
        self._save_checkpoint(checkpoint)
        return checkpoint
    
    def _save_checkpoint(self, checkpoint_data: Dict):
        """保存检查点"""
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"保存检查点失败: {e}")
    
    def _initialize_model(self):
        """初始化SD3模型管道"""
        if self.pipeline is not None:
            return
            
        self.logger.info("正在初始化模型...")
        
        # 设置随机种子
        def set_seed(seed: int = 42):
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True 
            torch.backends.cudnn.benchmark = False
        
        set_seed(42)
        
        # 检查模型是否存在
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"模型目录不存在: {self.model_dir}")
        
        # 配置量化
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # 加载模型
        try:
            model_nf4 = SD3Transformer2DModel.from_pretrained(
                self.model_dir,
                subfolder="transformer",
                quantization_config=nf4_config,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            
            self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                self.model_dir, 
                transformer=model_nf4,
                torch_dtype=torch.bfloat16
            )
            
            # 启用CPU卸载以节省显存[2](@ref)
            self.pipeline.enable_model_cpu_offload()
            
            # 初始化生成器
            self.generator = torch.Generator(device=self.device).manual_seed(42)
            
            self.logger.info("模型初始化完成")
            
        except Exception as e:
            self.logger.error(f"模型初始化失败: {e}")
            raise
    
    def _cleanup_memory(self):
        """清理内存和显存[6,7](@ref)"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        gc.collect()
        self.logger.debug("内存清理完成")
    
    def _generate_single_image(self, prompt: str, index: int) -> bool:
        """生成单张图片"""
        try:
            self.logger.info(f"正在生成第 {index} 张图片")
            
            # 生成图片[1](@ref)
            image = self.pipeline(
                prompt=prompt,
                height=512,
                width=512,
                generator=self.generator,
                num_images_per_prompt=1,
                num_inference_steps=30,
                guidance_scale=4.5,
                max_sequence_length=512,
            ).images[0]
            
            # 保存图片，使用6位数字命名[9](@ref)
            filename = f"{index:06d}.png"
            filepath = os.path.join(self.output_dir, filename)
            image.save(filepath)
            
            self.logger.info(f"图片已保存: {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"生成第 {index} 张图片失败: {e}")
            return False
    
    def load_dataset(self):
        """加载数据集"""
        self.logger.info("正在加载数据集...")
        try:
            self.dataset = load_from_disk(self.dataset_path)
            self.prompts = self.dataset['train']['Prompt'][:self.max_images]
            self.logger.info(f"数据集加载完成，共 {len(self.prompts)} 个提示词")
        except Exception as e:
            self.logger.error(f"数据集加载失败: {e}")
            raise
    
    def generate_batch(self, batch_size: int = 10) -> Dict[str, int]:
        """
        批量生成图片
        
        Args:
            batch_size: 每批处理数量，用于内存管理
            
        Returns:
            生成统计信息
        """
        if not hasattr(self, 'prompts'):
            self.load_dataset()
        
        if self.pipeline is None:
            self._initialize_model()
        
        # 从断点恢复
        start_index = self.checkpoint_data["last_index"] + 1
        if start_index >= len(self.prompts):
            self.logger.info("所有图片已生成完成")
            return self._get_generation_stats()
        
        if start_index > self.start_index:
            self.logger.info(f"从断点恢复，开始索引: {start_index}")
        
        # 设置开始时间
        if self.checkpoint_data["start_time"] is None:
            self.checkpoint_data["start_time"] = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            self.checkpoint_data["start_time"] = torch.cuda.Event(enable_timing=True)
            self.checkpoint_data["start_time"].record()
        
        self.logger.info(f"开始批量生成，从索引 {start_index} 到 {min(len(self.prompts), self.max_images) - 1}")
        
        successful_count = 0
        failed_count = 0
        
        for i in range(start_index, min(len(self.prompts), self.max_images)):
            try:
                prompt = self.prompts[i]
                
                # 生成单张图片
                success = self._generate_single_image(prompt, i)
                
                if success:
                    successful_count += 1
                    self.checkpoint_data["generated_indices"].append(i)
                else:
                    failed_count += 1
                    self.checkpoint_data["failed_indices"].append(i)
                
                # 更新检查点
                self.checkpoint_data["last_index"] = i
                self.checkpoint_data["total_generated"] = successful_count
                
                # 每batch_size张图片或最后一张图片时保存检查点并清理内存[8](@ref)
                if (i - start_index + 1) % batch_size == 0 or i == min(len(self.prompts), self.max_images) - 1:
                    self._save_checkpoint(self.checkpoint_data)
                    self._cleanup_memory()
                    self.logger.info(f"进度: {i - start_index + 1}/{min(len(self.prompts), self.max_images) - start_index} "
                                   f"(成功: {successful_count}, 失败: {failed_count})")
                
            except Exception as e:
                self.logger.error(f"处理索引 {i} 时发生异常: {e}")
                failed_count += 1
                self.checkpoint_data["failed_indices"].append(i)
                continue
        
        # 最终统计
        stats = self._get_generation_stats()
        self.logger.info(f"批量生成完成: {stats}")
        
        return stats
    
    def _get_generation_stats(self) -> Dict[str, int]:
        """获取生成统计信息"""
        return {
            "total_generated": len(self.checkpoint_data["generated_indices"]),
            "total_failed": len(self.checkpoint_data["failed_indices"]),
            "completion_rate": len(self.checkpoint_data["generated_indices"]) / min(len(self.prompts), self.max_images) * 100
        }
    
    def get_failed_indices(self) -> List[int]:
        """获取失败的索引列表"""
        return self.checkpoint_data.get("failed_indices", [])
    
    def retry_failed(self, batch_size: int = 5) -> Dict[str, int]:
        """重试失败的生成任务"""
        failed_indices = self.get_failed_indices()
        if not failed_indices:
            self.logger.info("没有失败的生成任务需要重试")
            return self._get_generation_stats()
        
        self.logger.info(f"开始重试 {len(failed_indices)} 个失败任务")
        
        # 临时保存原始失败列表
        original_failed = failed_indices.copy()
        self.checkpoint_data["failed_indices"] = []
        
        successful_retry = 0
        
        for i, index in enumerate(original_failed):
            try:
                prompt = self.prompts[index]
                success = self._generate_single_image(prompt, index)
                
                if success:
                    successful_retry += 1
                    self.checkpoint_data["generated_indices"].append(index)
                else:
                    self.checkpoint_data["failed_indices"].append(index)
                
                # 更新统计
                self.checkpoint_data["total_generated"] = len(self.checkpoint_data["generated_indices"])
                
                # 定期保存检查点
                if (i + 1) % batch_size == 0:
                    self._save_checkpoint(self.checkpoint_data)
                    self._cleanup_memory()
                    
            except Exception as e:
                self.logger.error(f"重试索引 {index} 失败: {e}")
                self.checkpoint_data["failed_indices"].append(index)
                continue
        
        self._save_checkpoint(self.checkpoint_data)
        self.logger.info(f"重试完成: 成功 {successful_retry}/{len(original_failed)}")
        
        return self._get_generation_stats()
