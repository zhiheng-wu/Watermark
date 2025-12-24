import os
import json
import subprocess
import random
from abc import ABC, abstractmethod
from pathlib import Path

# ============================
# 1. 基类定义：增加采样率逻辑
# ============================
class BaseWatermarker(ABC):
    def __init__(self, sampling_ratio: float = 1.0, seed: int = 42, **kwargs):
        """
        :param sampling_ratio: 采样率 (0.0 - 1.0)，默认 1.0 (全量)
        :param seed: 随机种子，保证每次采样文件一致
        :param kwargs: 其他算法参数
        """
        self.sampling_ratio = sampling_ratio
        self.seed = seed
        self.params = kwargs
        # 将采样参数合并进 params，以便后续 Proxy 传递
        self.params['sampling_ratio'] = sampling_ratio
        self.params['seed'] = seed

    def _filter_files(self, input_dir: str, extensions={'.jpg', '.png', '.jpeg', '.bmp'}):
        """
        通用工具方法：获取目录下所有图片，并根据采样率进行筛选
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            return []
            
        all_files = [
            f for f in input_path.iterdir() 
            if f.is_file() and f.suffix.lower() in extensions
        ]
        
        # 排序以确保随机数种子的稳定性
        all_files.sort()
        
        # 如果是全量，直接返回
        if self.sampling_ratio >= 1.0:
            return all_files
            
        # 根据比率计算数量
        num_samples = int(len(all_files) * self.sampling_ratio)
        if num_samples < 1 and len(all_files) > 0:
            num_samples = 1 # 至少处理一张
            
        random.seed(self.seed)
        selected_files = random.sample(all_files, num_samples)
        
        print(f"[Info] Sampling enabled: {len(selected_files)}/{len(all_files)} files (Ratio: {self.sampling_ratio})")
        return selected_files

    @abstractmethod
    def process(self, input_dir: str, output_dir: str): 
        """
        子类需实现具体的加/解水印逻辑
        建议子类调用 self._filter_files(input_dir) 获取待处理文件列表
        """
        pass

# ============================
# 2. 代理类：跨环境调用
# ============================
class CrossEnvWatermarkerProxy(BaseWatermarker):
    def __init__(self, method_name, target_env, **kwargs):
        # 初始化基类
        super().__init__(**kwargs)
        self.method_name = method_name
        self.target_env = target_env
        # self.params 在基类中已经包含了 sampling_ratio

    def process(self, input_dir: str, output_dir: str):
        # 获取 worker.py 的绝对路径 (假设和 core.py 在同一级，或者你需要指定绝对路径)
        current_dir = Path(__file__).parent.absolute()
        worker_script = current_dir / "worker.py"
        
        # 准备参数
        params_json = json.dumps(self.params)

        # 你的 Conda 路径配置
        conda_base = Path(r"D:/ProgramSoftware/anaconda3/envs")
        target_python = conda_base / self.target_env / "python.exe"
        
        # 构建命令
        cmd = [
            str(target_python), "-u", str(worker_script),
            "--name", self.method_name,
            "--input", str(input_dir),
            "--output", str(output_dir),
            "--params", params_json
        ]

        print(f"[Factory] Switching to env '{self.target_env}' to run {self.method_name} (Ratio: {self.sampling_ratio})...")
        
        try:
            print(f"[Factory] >>> Subprocess Output ({self.target_env}) >>>")
            subprocess.run(cmd, check=True, text=True) 
            print(f"[Factory] <<< Subprocess Finished <<<")
        except subprocess.CalledProcessError as e:
            print(f"[Factory] Subprocess failed with exit code {e.returncode}")
            raise e

# ============================
# 3. 工厂类
# ============================
class WatermarkerFactory:
    _registry = {}

    @classmethod
    def register(cls, name: str, conda_env: str = None):
        def decorator(subclass):
            cls._registry[name] = {
                "class": subclass,
                "env": conda_env
            }
            return subclass
        return decorator

    @classmethod
    def create(cls, name, params=None, ignore_env_check=False):
        if name not in cls._registry:
            raise ValueError(f"Watermarker '{name}' not found. Available: {list(cls._registry.keys())}")
        
        entry = cls._registry[name]
        target_env = entry["env"]
        cls_ref = entry["class"]
        params = params or {}

        # 检查环境
        current_env = os.environ.get("CONDA_DEFAULT_ENV", None)
        
        # 判断逻辑：如果不忽略检查，且指定了环境，且环境不匹配 -> 返回代理
        if not ignore_env_check and target_env and target_env != current_env:
            # Proxy 会自动处理 params 里的 sampling_ratio
            return CrossEnvWatermarkerProxy(name, target_env, **params)
        else:
            # 本地直接实例化
            return cls_ref(**params)