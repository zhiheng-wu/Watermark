import os
from pathlib import Path
from PIL import Image
from attacks.core import BaseAttacker, AttackerFactory
import sys

@AttackerFactory.register("ctrl_regen", conda_env="ctrl")
class CtrlRegenAttacker(BaseAttacker):
    """
    基于 ControlNet 重绘的攻击 (CtrlRegen)。
    
    注意：
    1. 指定了 conda_env="ctrl"，工厂会自动通过 worker.py 在该环境中运行。
    2. 依赖项 'models.attackers...' 必须在 process 内部导入，防止污染主环境。
    """
    def process(self, input_dir: str, output_dir: str):
        # -----------------------------------------------------------
        # 关键点：延迟导入
        # 只有在 worker 进程切换到 'ctrl' 环境后，这行代码才会被执行。
        # 如果放在文件开头，主程序扫描注册表时会因为缺少依赖报错。
        # -----------------------------------------------------------
        try:
            # 假设 worker.py 已经将项目根目录加入 sys.path，
            # 这里可以直接 import 同级的 models 模块
            sys.path.append("D:/graduation/computer/Watermark/models/attackers/CtrlRegen")
            from models.attackers.CtrlRegen.ctrl import ctrl_regen_plus
        except ImportError as e:
            print(f"[CtrlRegen] Import Error: {e}")
            print("[CtrlRegen] Make sure you are running in the 'ctrl' conda environment.")
            print("[CtrlRegen] Check if project root is in PYTHONPATH.")
            raise e

        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        step = self.params.get('step', 0.4)
        seed = self.params.get('seed', 42)
        
        print(f"[CtrlRegen] Starting generation with step={step}, seed={seed}...")

        count = 0
        for img_file in input_path.iterdir():
            if img_file.suffix.lower() not in ['.jpg', '.png', '.jpeg', '.bmp']:
                continue
            
            try:
                # 1. 读取原始图片
                # Diffusers 管道通常期望 PIL Image 输入
                original_img = Image.open(img_file).convert("RGB")
                W, H = original_img.size
                
                # 2. 调用外部攻击模型
                # 注意：ctrl_regen_plus 内部可能会进行 resize (如 512x512)
                # 这一步通常比较耗时 (GPU 推理)
                attacked_pil = ctrl_regen_plus(original_img, step=step, seed=seed)
                
                # 3. 尺寸一致性检查与恢复
                # 攻击不仅要改变像素，通常要求保持分辨率不变以便后续处理
                if attacked_pil.size != (W, H):
                    attacked_pil = attacked_pil.resize((W, H), resample=Image.BILINEAR)
                
                # 4. 保存结果
                # 建议使用 PNG 保存以保留生成的细节，避免 JPEG 二次压缩
                attacked_pil.save(output_path / img_file.name, format='PNG')
                
                count += 1
                # 可选：打印进度，因为扩散模型很慢
                print(f"[CtrlRegen] Processed {img_file.name}")

            except Exception as e:
                print(f"[CtrlRegen] Error processing {img_file.name}: {e}")
                import traceback
                traceback.print_exc()

        print(f"[CtrlRegen] Completed. {count} images processed.")