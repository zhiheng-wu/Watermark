import argparse
import json
import sys
import pkgutil
import importlib
from pathlib import Path
import traceback

# ============== 配置路径 ================
sys.path.append("D:/graduation/computer/Watermark")

import watermarks.implementations as implementations
from watermarks.core import WatermarkerFactory

# package = implementations
# path = package.__path__
# prefix = package.__name__ + "."
# for _, name, is_pkg in pkgutil.iter_modules(path, prefix):
#     try:
#         # 尝试导入具体模块
#         importlib.import_module(name)
#         # print(f"[Worker] Imported {name}") 
#     except Exception as e:
#         # 关键修改：打印具体的堆栈信息，不要只打印 e
#         print(f"[Worker] ❌ Failed to import {name}")
#         print("-" * 20)
#         traceback.print_exc() 
#         print("-" * 20)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--mode", required=True, choices=['embed', 'extract'])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=False)
    # 新增参数：纯净原图来源
    parser.add_argument("--clean_source", required=False, help="Source of clean images for FPR test")
    parser.add_argument("--params", required=True)
    args = parser.parse_args()

    try:
        params = json.loads(args.params)
    except:
        sys.exit(1)
    
    try:
        importlib.import_module(f"watermarks.implementations.{args.name}")
        print(f"[Worker] Imported {args.name}") 
    except Exception as e:
        print(f"[Worker] ❌ Failed to import implementation for {args.name}")
        print("-" * 20)
        traceback.print_exc() 
        print("-" * 20)
        sys.exit(1)
    # 实例化
    watermarker = WatermarkerFactory.create(args.name, params, ignore_env_check=True)

    if args.mode == 'embed':
        if not args.output:
            raise ValueError("Embed mode requires --output")
        watermarker.embed_batch(args.input, args.output)
        
    elif args.mode == 'extract':
        threshold = params.get('threshold', 0.1)
        mix_ratio = params.get('mix_ratio', 1.0)
        
        # 调用更新后的 extract_batch
        # args.input -> 水印图目录
        # args.clean_source -> 原图目录
        results = watermarker.extract_batch(
            image_dir_to_check=args.input,
            threshold=threshold,
            clean_dir=args.clean_source,
            mix_ratio=mix_ratio
        )
        
        # print("\n<<<JSON_START>>>")
        # print(json.dumps(results))
        # print("<<<JSON_END>>>")

if __name__ == "__main__":
    main()