import shutil
from pathlib import Path
from attacks.core import BaseAttacker, AttackerFactory

@AttackerFactory.register("identity")
class IdentityAttacker(BaseAttacker):
    """
    恒等变换 (Identity)。
    用于测试 Pipeline 的连通性或作为无攻击对照组。
    策略：直接将输入文件夹的图像复制到输出文件夹，不进行任何解码/编码操作，确保绝对无损。
    """
    def process(self, input_dir: str, output_dir: str):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 定义需要处理的图片扩展名
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        print(f"[Identity] Copying files from {input_dir} to {output_dir}...")

        count = 0
        for img_file in input_path.iterdir():
            # 过滤非图片文件
            if img_file.suffix.lower() not in extensions:
                continue
            
            try:
                # 直接复制文件 (shutil.copy2 会保留文件元数据)
                shutil.copy2(img_file, output_path / img_file.name)
                count += 1
            except Exception as e:
                print(f"[Identity] Error processing {img_file.name}: {e}")
        
        print(f"[Identity] Processed {count} images.")