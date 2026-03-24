import os
import json
import subprocess
import random
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Tuple

class BaseWatermarker(ABC):
    def __init__(self, sampling_ratio: float = 0.05, seed: int = 42, **kwargs):
        self.sampling_ratio = sampling_ratio
        self.seed = seed
        self.params = kwargs
        
        # 确保参数传递
        self.params['sampling_ratio'] = sampling_ratio
        self.params['seed'] = seed
        
        # 内置水印内容 (Secret)
        self.secret = kwargs.get('secret', [1, 0, 1, 0]) 

    def _filter_files_with_sampling(self, input_dir: str) -> List[Path]:
        """ 仅用于【嵌入】阶段：根据采样率筛选源文件 """
        input_path = Path(input_dir)
        if not input_path.exists(): return []
        
        all_files = sorted([
            f for f in input_path.iterdir() 
            if f.is_file() and f.suffix.lower() in {'.jpg', '.png', '.bmp', '.jpeg'}
        ])
        
        if self.sampling_ratio >= 1.0: 
            return all_files
        
        random.seed(self.seed)
        num_samples = max(1, int(len(all_files) * self.sampling_ratio))
        print(f"[Info] Sampling: {num_samples}/{len(all_files)} images (Ratio: {self.sampling_ratio})")
        return random.sample(all_files, num_samples)

    def _build_mixed_test_set(self, watermark_dir: Path, clean_dir: Path, ratio: float) -> List[Tuple[Path, bool]]:
        """
        构建混合测试集
        :param watermark_dir: 存放已嵌入水印图片的目录
        :param clean_dir: 存放原图的目录
        :param ratio: 测试集中【含水印图片】的占比 (0.0 ~ 1.0)
        :return: List[(图片路径, 是否含水印)]
        """
        # 1. 获取所有含水印图片
        wm_files = sorted([f for f in watermark_dir.iterdir() if f.is_file()])
        if not wm_files:
            return []

        # 如果没有提供 clean_dir 或者 比例为 1.0，则全测水印图
        if clean_dir is None or ratio >= 1.0:
            return [(f, True) for f in wm_files]

        # 2. 计算需要多少张 clean 图片
        # 设 total 为总数，num_wm 为水印图数量。
        # 策略：尽量使用所有生成好的水印图，以此推算需要补充多少原图。
        # num_wm = total * ratio  ==> total = num_wm / ratio
        # num_clean = total - num_wm
        
        num_wm = len(wm_files)
        if ratio <= 0: # 极端情况
            total_needed = num_wm # 这里的逻辑看需求，这里简单处理
        else:
            total_needed = int(num_wm / ratio)
        
        num_clean_needed = total_needed - num_wm
        
        # 3. 从 clean_dir 获取原图，注意：为了避免文件名混淆，
        # 通常可以选取那些【不在】wm_files 里的图，或者单纯随机选取。
        # 为了更严格的假阳性测试（FPR），我们最好选取那些【未被嵌入】的图片。
        # 但既然文件名一致，我们也可以选取同名图片（这就变成了攻击鲁棒性测试），
        # 这里假设用户想要的是“未含水印的样本”。
        
        clean_files_all = sorted([
            f for f in clean_dir.iterdir() 
            if f.is_file() and f.suffix.lower() in {'.jpg', '.png', '.bmp', '.jpeg'}
        ])
        
        # 简单策略：随机抽取 needed 数量的 clean 图片
        # (注：实际场景可能需要排除掉对应的 wm_files 同名文件，取决于你想测什么)
        # 这里我们做简单随机抽样
        random.seed(self.seed)
        if len(clean_files_all) > num_clean_needed:
            clean_samples = random.sample(clean_files_all, num_clean_needed)
        else:
            clean_samples = clean_files_all # 不够就全上
            
        # 4. 组装
        dataset = [(f, True) for f in wm_files] + [(f, False) for f in clean_samples]
        random.shuffle(dataset) # 打乱顺序
        return dataset

    # ===========================
    # 模式 B: 提取 (混合检测)
    # ===========================
    def extract_batch(self, 
                      image_dir_to_check: str, 
                      threshold: float, 
                      clean_dir: str = None, 
                      mix_ratio: float = 1.0) -> Dict[str, Any]:
        """
        :param image_dir_to_check: 含水印图片的目录 (Positive Samples)
        :param clean_dir: 原图目录 (Negative Samples)，用于测试假阳性
        :param mix_ratio: 含水印图片在总测试集中的比例
        """
        wm_path = Path(image_dir_to_check)
        cl_path = Path(clean_dir) if clean_dir else None
        
        # 构建任务列表： [(path, is_watermarked_ground_truth), ...]
        tasks = self._build_mixed_test_set(wm_path, cl_path, mix_ratio)
        
        results = {
            "total": len(tasks),
            "metrics": {
                "TP": 0, "FP": 0, "TN": 0, "FN": 0
            },
            "accuracy": 0.0,
            "fpr": 0.0, # False Positive Rate
            "details": {}
        }

        print(f"[{self.__class__.__name__}] Extraction start. Checking {len(tasks)} files (Mix Ratio: {mix_ratio})")
        ji=0
        for f_path, is_wm_gt in tasks:
            try:
                extracted_data = self.extract_one(f_path)
                distance = self.calculate_distance(self.secret, extracted_data)
                # 判定
                is_detected = distance < threshold
                
                # 记录详情
                file_key = f"{'WM' if is_wm_gt else 'CLN'}_{f_path.name}"
                results["details"][file_key] = {
                    "distance": float(distance),
                    "is_watermarked_gt": is_wm_gt, # Ground Truth
                    "detected": is_detected
                }
                
                # 统计混淆矩阵
                if is_wm_gt:
                    if is_detected: results["metrics"]["TP"] += 1
                    else:           results["metrics"]["FN"] += 1
                else:
                    if is_detected: results["metrics"]["FP"] += 1
                    else:           results["metrics"]["TN"] += 1

            except Exception as e:
                print(f"  [Extract Error] {f_path.name}: {e}")
                # 出错视作未检测到
                if is_wm_gt: results["metrics"]["FN"] += 1
                else:        results["metrics"]["TN"] += 1 # 没报错也没检测到，算TN? 或者忽略
        # 计算最终指标
        m = results["metrics"]
        # Accuracy
        if results["total"] > 0:
            results["accuracy"] = (m["TP"] + m["TN"]) / results["total"]
        
        # False Positive Rate = FP / (FP + TN)
        neg_total = m["FP"] + m["TN"]
        if neg_total > 0:
            results["fpr"] = m["FP"] / neg_total
        else:
            results["fpr"] = 0.0 # 无负样本

        print(f"[{self.__class__.__name__}] Done. Acc: {results['accuracy']:.2%} | FPR: {results['fpr']:.2%}")
        print(f"Stats: TP={m['TP']}, FN={m['FN']}, FP={m['FP']}, TN={m['TN']}")
        
        return results

    # embed_batch, embed_one, extract_one, calculate_distance 保持不变...
    def embed_batch(self, input_dir: str, output_dir: str):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        files = self._filter_files_with_sampling(input_dir)
        print(f"[{self.__class__.__name__}] Embedding start. Processing {len(files)} files...")
        for f in files:
            try:
                self.embed_one(f, output_path / f.name)
            except Exception as e:
                print(f"  [Embed Fail] {f.name}: {e}")

    @abstractmethod
    def embed_one(self, src_path: Path, dst_path: Path): pass

    @abstractmethod
    def extract_one(self, src_path: Path): pass

    def calculate_distance(self, original, extracted):
        arr_org = np.array(original).flatten()
        arr_ext = np.array(extracted).flatten()
        if len(arr_org) != len(arr_ext): return 1.0
        return np.mean(arr_org != arr_ext)


# ============================
# 代理类
# ============================
class CrossEnvProxy(BaseWatermarker):
    def __init__(self, method_name, target_env, **kwargs):
        super().__init__(**kwargs)
        self.method_name = method_name
        self.target_env = target_env

    def embed_batch(self, input_dir, output_dir):
        self._run_worker("embed", input_path=input_dir, output_path=output_dir)

    def extract_batch(self, image_dir_to_check, threshold, clean_dir=None, mix_ratio=1.0):
        # 将新增参数放入 params 传给 worker
        self.params['threshold'] = threshold
        self.params['mix_ratio'] = mix_ratio
        
        # clean_dir 需要作为路径参数传递
        clean_source_arg = str(clean_dir) if clean_dir else None
        
        self._run_worker("extract", 
                         input_path=image_dir_to_check, 
                         clean_source=clean_source_arg)
        return {} # 依然由 Worker 打印 JSON

    def _run_worker(self, mode, input_path, output_path=None, clean_source=None):
        current_dir = Path(__file__).parent.absolute()
        worker_script = current_dir / "worker.py"
        params_json = json.dumps(self.params)
        
        conda_base = Path(r"D:/ProgramSoftware/anaconda3/envs")
        target_python = conda_base / self.target_env / "python.exe"
        
        cmd = [
            str(target_python), "-u", str(worker_script),
            "--name", self.method_name,
            "--mode", mode,
            "--input", str(input_path),
            "--params", params_json
        ]
        
        if output_path:
            cmd.extend(["--output", str(output_path)])
            
        # 传递原图路径用于混合测试
        if clean_source:
            cmd.extend(["--clean_source", str(clean_source)])

        print(f"[Proxy] Calling {self.method_name} [{mode}]...")
        print(f"[Proxy] Command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, text=True)
    
    def embed_one(self, src_path: Path, dst_path: Path):
        raise NotImplementedError("CrossEnvProxy does not support embed_one directly.")
    
    def extract_one(self, src_path: Path):
        raise NotImplementedError("CrossEnvProxy does not support extract_one directly.")

# Factory 
class WatermarkerFactory:
    _registry = {}

    @classmethod
    def register(cls, name: str, conda_env: str = None):
        def decorator(subclass):
            cls._registry[name] = {"class": subclass, "env": conda_env}
            return subclass
        return decorator

    @classmethod
    def create(cls, name, params=None, ignore_env_check=False):
        if name not in cls._registry: raise ValueError(f"Unknown: {name}")
        entry = cls._registry[name]
        current_env = os.environ.get("CONDA_DEFAULT_ENV", None)
        if not ignore_env_check and entry["env"] and entry["env"] != current_env:
            return CrossEnvProxy(name, entry["env"], **(params or {}))
        return entry["class"](**(params or {}))