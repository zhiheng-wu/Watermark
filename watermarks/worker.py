import argparse
import json
import sys
import os
import importlib
import pkgutil
from pathlib import Path

# ============== 配置路径 ================
# 确保能够导入项目根目录
PROJECT_ROOT = "D:/graduation/computer/Watermark" 
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 假设 core.py 在 watermarks 文件夹下
from watermarks.core import WatermarkerFactory
import watermarks.implementations as implementations # 导入具体的实现包

# 动态加载所有实现，触发 @register
def load_plugins(package):
    prefix = package.__name__ + "."
    for _, name, is_pkg in pkgutil.iter_modules(package.__path__, prefix):
        try:
            importlib.import_module(name)
        except ImportError as e:
            print(f"[Worker] Warning: Failed to import {name}: {e}")

try:
    load_plugins(implementations)
except Exception:
    pass # 如果没有子包结构，可以直接在 implementations/__init__.py 里 import

def main():
    parser = argparse.ArgumentParser(description="Watermark Worker Process")
    parser.add_argument("--name", required=True, help="Registered method name")
    parser.add_argument("--input", required=True, help="Input directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--params", required=True, help="JSON params string")

    args = parser.parse_args()

    try:
        params = json.loads(args.params)
    except json.JSONDecodeError:
        print("[Worker] Error: Params JSON decode failed")
        sys.exit(1)

    current_env = os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
    print(f"[Worker] Initializing '{args.name}' in env: {current_env}")

    try:
        # 1. 创建实例 (ignore_env_check=True 强制返回真实对象)
        # 注意：params 中包含了 sampling_ratio，会被传入 BaseWatermarker 的 __init__
        watermarker = WatermarkerFactory.create(args.name, params, ignore_env_check=True)
        
        # 2. 执行处理
        watermarker.process(args.input, args.output)
        
        print("[Worker] Task completed successfully.")
        
    except Exception as e:
        print(f"[Worker] Critical Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()