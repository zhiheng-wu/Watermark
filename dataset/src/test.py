
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers import StableDiffusion3Pipeline
import torch
import os
# ==================== 1. 固定所有随机种子 ====================
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 

# 设置种子
set_seed(42)

# ==================== 2. 固定 generator ====================
generator = torch.Generator(device="cuda").manual_seed(42)

# 指定本地模型路径
model_dir = "D:/graduation/computer/Watermark/models/sd3.5"  # 确保这是你实际下载模型保存的路径

# 检查模型是否存在
if not os.path.exists(model_dir):
    raise FileNotFoundError(f"模型目录不存在: {model_dir}")
print(f"模型目录找到: {model_dir}")
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
print("正在加载量化模型，请稍候...")
model_nf4 = SD3Transformer2DModel.from_pretrained(
    model_dir,
    subfolder="transformer",
    quantization_config=nf4_config,
    torch_dtype=torch.bfloat16
)
print("量化模型加载完成。")
print("正在创建Stable Diffusion 3管道，请稍候...")
pipeline = StableDiffusion3Pipeline.from_pretrained(
    model_dir, 
    transformer=model_nf4,
    torch_dtype=torch.bfloat16
)
pipeline.enable_model_cpu_offload()
print("Stable Diffusion 3管道创建完成。")

from datasets import load_dataset, load_from_disk

# 从本地磁盘加载已保存的数据集
dataset = load_from_disk("D:/graduation/computer/Watermark/dataset/prompts/stable_diffusion_prompts")
prompt = dataset['train'][0]['Prompt']
print(f"使用的提示词: {prompt}")
image = pipeline(
    prompt=prompt,
    height=512,
    width=512,
    generator=generator,
    num_images_per_prompt=1,
    num_inference_steps=30,
    guidance_scale=4.5,
    max_sequence_length=512,
).images[0]
image.save("../test/p1.png")
print("图片已保存为 p1.png")