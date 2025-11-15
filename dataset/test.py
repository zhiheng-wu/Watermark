
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers import StableDiffusion3Pipeline
import torch
import os

# 指定本地模型路径
model_dir = "../models/sd3.5/"  # 确保这是你实际下载模型保存的路径

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

prompt = ""
image = pipeline(
    prompt=prompt,
    seed=55,
    height=512,
    width=512,
    num_images_per_prompt=1,
    num_inference_steps=30,
    guidance_scale=4.5,
    max_sequence_length=512,
).images[0]
image.save("./test/generated_image.png")
print("图片已保存为 generated_image.png")