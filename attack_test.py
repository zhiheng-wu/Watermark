from attacks.core import AttackerFactory
import importlib
import attacks.implementations
import pkgutil

imported_modules = {}
package = attacks.implementations
prefix = package.__name__ + "."
for _, name, is_pkg in pkgutil.iter_modules(package.__path__, prefix):
    # 这里的 name 已经是完整的路径，例如 'attacks.imp.module_a'
    module = importlib.import_module(name)
    
    # 获取短模块名 (例如 'module_a') 作为 key
    short_name = name.split('.')[-1]
    imported_modules[short_name] = module
    
    print(f"成功导入: {short_name}")

def run_pipeline():
    input_folder = "./dataset/origin"
    output_folder = "./dataset/test"
    
    # 1. 调用本地环境的攻击
    attacker1 = AttackerFactory.create_attacker("identity")

    print("Starting cross-env attack...")
    attacker2 = AttackerFactory.create_attacker("ctrl_regen")
    attacker2.process(input_folder, output_folder)
    print("Pipeline finished.")

if __name__ == "__main__":
    run_pipeline()