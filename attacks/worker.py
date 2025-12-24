import argparse
import json
import sys
import os
from pathlib import Path
import importlib
import implementations
import pkgutil

# 这一步是为了确保能 import 到 attacks 模块
# 假设 output/attacks/worker.py 被执行，我们将项目根目录加入 sys.path
sys.path.append("D:/graduation/computer/Watermark")

from attacks.core import AttackerFactory
# 重要：这里必须导入定义攻击方法的模块，触发 @register 装饰器
# 实际项目中，你可能需要根据配置动态导入，或者简单的 import 所有实现
try:
    package = implementations
    prefix = package.__name__ + "."
    for _, name, is_pkg in pkgutil.iter_modules(package.__path__, prefix):
        module = importlib.import_module(name)
    print("子模块导入成功")
except ImportError:
    pass # 具体根据你的模块结构调整

def main():
    parser = argparse.ArgumentParser(description="Worker process for specific conda env")
    parser.add_argument("--name", required=True, help="Registered attacker name")
    parser.add_argument("--input", required=True, help="Input directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--params", required=True, help="JSON string of params")

    args = parser.parse_args()

    # 解析参数
    try:
        params = json.loads(args.params)
    except json.JSONDecodeError:
        print("Error: Params is not a valid JSON string")
        sys.exit(1)

    print(f"[Worker] Running {args.name} in env: {os.environ.get('CONDA_DEFAULT_ENV')}")

    try:
        # 此时 create_attacker 看到的 "current_env" 就是目标环境，
        # 所以它会直接返回真实的类实例，而不是 Proxy。
        attacker = AttackerFactory.create_attacker(args.name, params, ignore_env_check=True)
        attacker.process(args.input, args.output)
        print("[Worker] Processing finished.")
    except Exception as e:
        print(f"[Worker] Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()