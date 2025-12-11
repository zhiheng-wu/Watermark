# attacks/__init__.py
import os
import glob
import importlib

# 获取当前文件夹路径
current_dir = os.path.dirname(__file__)

# 查找所有 .py 文件
modules = glob.glob(os.path.join(current_dir, "*.py"))

# 遍历并导入所有模块 (排除 __init__.py)
for f in modules:
    if os.path.isfile(f) and not f.endswith('__init__.py'):
        module_name = os.path.basename(f)[:-3] # 去掉 .py
        # 动态导入: . 表示当前包
        importlib.import_module(f".{module_name}", package=__name__)

# 暴露工厂类
from .core import AttackerFactory