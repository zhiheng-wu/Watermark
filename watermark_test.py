from watermarks.core import WatermarkerFactory
import watermarks.implementations
import importlib
import pkgutil
from utils.monitor import WatermarkProfiler # type: ignore

imported_modules = {}
package = watermarks.implementations
prefix = package.__name__ + "."
# ============== 动态加载所有实现 ================
for _, name, is_pkg in pkgutil.iter_modules(package.__path__, prefix):
    # 这里的 name 已经是完整的路径，例如 'attacks.imp.module_a'
    module = importlib.import_module(name)
    
    # 获取短模块名 (例如 'module_a') 作为 key
    short_name = name.split('.')[-1]
    imported_modules[short_name] = module
    
    print(f"成功导入: {short_name}")

# ============= 配置实验参数 ================
# common parameters
sampling_ratio = 0.002  # 0.002% 的图片
seed = 42  # 固定随机种子
mix_ratio = 0.5
input_folder = "D:/graduation/computer/Watermark/dataset/origin"
output_folder = "D:/graduation/computer/Watermark/dataset/watermarked"
threshold = 0.05
prompt_input_folder = 'D:/graduation/computer/Watermark/dataset/prompts/stable_diffusion_prompts'
# mbrs
secret_64 = [1,0,1,0, 0,1,0,1, 1,1,0,0, 0,0,1,1,
             1,0,0,1, 0,1,1,0, 1,1,1,0, 0,0,0,1,
             0,1,0,0, 1,0,1,1, 1,0,0,0, 0,1,1,1,
             1,1,0,1, 0,0,1,0, 1,0,1,0, 0,1,0,1]  # 64 bits
# cin
secret_30 = [1,0,1,0, 0,1,0,1, 1,1,0,0, 0,0,1,1,
             1,0,0,1, 0,1,1,0, 1,1,1,0, 0,0]
# ============== 测试水印嵌入与提取 ================
def run_experiment():
    attacker = WatermarkerFactory.create(
        name="tree_ring",
        params={
            'image_length': 512,
            'seed': seed,
            'max_images':10
        }
    )

    
    # 这一步会自动检测环境，如果不匹配则调用 worker.py
    with WatermarkProfiler(str(attacker.__class__.__name__)+'_embedding'):
        attacker.embed_batch(prompt_input_folder, output_folder)
    with WatermarkProfiler(str(attacker.__class__.__name__)+'_extraction'):
        result = attacker.extract_batch(
            image_dir_to_check=output_folder,
            clean_dir=input_folder,
            threshold=threshold,
            mix_ratio = mix_ratio
        )

if __name__ == "__main__":
    run_experiment()