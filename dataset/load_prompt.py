from datasets import load_dataset

ds = load_dataset("Gustavosta/Stable-Diffusion-Prompts")
ds.save_to_disk("prompts/stable_diffusion_prompts")
print("数据集已保存到 dataset/stable_diffusion_prompts")