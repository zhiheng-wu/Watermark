import os
import json
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path

# 定义基类：改为面向文件流的处理
class BaseAttacker(ABC):
    def __init__(self, **kwargs):
        self.params = kwargs
    
    @abstractmethod
    def process(self, input_dir: str, output_dir: str): 
        """
        核心处理逻辑：读取input_dir中的图片，处理后存入output_dir，保持文件名一致
        """
        pass

# 代理攻击者：用于在不同环境中运行真正的攻击者
class CrossEnvAttackerProxy(BaseAttacker):
    def __init__(self, attack_name, target_env, **kwargs):
        super().__init__(**kwargs)
        self.attack_name = attack_name
        self.target_env = target_env
        self.params = kwargs

    def process(self, input_dir: str, output_dir: str):
        # 获取 worker.py 的绝对路径
        current_dir = Path(__file__).parent.absolute()
        worker_script = current_dir / "worker.py"
        
        # 准备参数 (将字典序列化为 JSON 字符串传递)
        params_json = json.dumps(self.params)

        conda_base = Path(r"D:/ProgramSoftware/anaconda3/envs")
        target_python = conda_base / self.target_env / "python.exe"
        
        # 构建 Conda 命令
        cmd=[
            str(target_python), "-u", str(worker_script),
            "--name", self.attack_name,
            "--input", str(input_dir),
            "--output", str(output_dir),
            "--params", params_json
        ]

        print(f"[Factory] Switching to env '{self.target_env}' to run {self.attack_name}...")
        
        # 同步执行子进程
        try:
            print(f"[Factory] Subprocess output start ({self.target_env}) >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            subprocess.run(cmd, check=True, text=True) 
            print(f"[Factory] Subprocess output end <<<<<<<<<<<<<<<<<<<<<<<<<<<")
        except subprocess.CalledProcessError as e:
            print(f"[Factory] Subprocess failed with exit code {e.returncode}")
            raise e

class AttackerFactory:
    _registry = {}

    @classmethod
    def register(cls, name: str, conda_env: str = None):
        """
        装饰器：注册攻击方法
        :param name: 攻击方法名称
        :param conda_env: 指定该方法运行的 conda 环境名称。如果不填，默认在调用者的环境中运行。
        """
        def decorator(subclass):
            cls._registry[name] = {
                "class": subclass,
                "env": conda_env
            }
            return subclass
        return decorator

    @classmethod
    def create_attacker(cls, name, params=None, ignore_env_check=False):
        if name not in cls._registry:
            raise ValueError(f"Attacker '{name}' not found. Registered: {list(cls._registry.keys())}")
        
        entry = cls._registry[name]
        target_env = entry["env"]
        attacker_class = entry["class"]
        params = params or {}

        # 判断是否需要跨环境
        current_env = os.environ.get("CONDA_DEFAULT_ENV", None)
        
        # 如果指定了环境，且与当前环境不同，则返回代理
        # 注意：如果 target_env 为 None，意味着可以在任何环境运行（或当前环境）
        if not ignore_env_check and target_env and target_env != current_env:
            return CrossEnvAttackerProxy(name, target_env, **params)
        else:
            # 直接返回真实的类实例
            return attacker_class(**params)