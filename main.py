
import os
import warnings
import logging

# 1. 屏蔽 Python 标准警告 (针对 timm, controlnet_aux 等库产生的 FutureWarning/UserWarning)
warnings.filterwarnings("ignore")

# 2. 屏蔽 Hugging Face 库的普通日志 (针对 transformers 和 diffusers)
from transformers import logging as t_logging
t_logging.set_verbosity_error()

from diffusers import logging as d_logging
d_logging.set_verbosity_error()
import sys
pths = ["D:/graduation/computer/Watermark/attacks", "D:/graduation/computer/Watermark/models/attackers/CtrlRegen"]
sys.path.extend(pths)
import torch
from attacks.core import AttackerFactory
import matplotlib.pyplot as plt
import torchvision


# 使用matplotlib的imshow显示和保存
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    print(f"CUDA可用，将在 {device} 上测试。")

    # 取一张测试图像 (C, H, W 格式，转换为浮点数，放在 GPU 上)
    gpu_image = torchvision.io.read_image("D:/graduation/computer/Watermark/dataset/origin/000000.png").float()/255.0
    gpu_image=gpu_image.to(device)
    print(f"输入张量设备: {gpu_image.device}")

    # 实例化 GPU 攻击者
    print("实例化 GPU 攻击者...")
    # gpu_scaler = AttackerFactory.create_attacker("ctrl_regen", params={})
    attacker = AttackerFactory.create_attacker(
        "pattern_estimation", 
        params={
            "clean_path": "./dataset/origin",       # 替换为实际路径
            "watermarked_path": "./dataset/origin",    # 替换为实际路径
            "num_images": 100
        }
    )
    print("GPU 攻击者实例化完成。")
    
    # 执行攻击
    print("开始 GPU 攻击...")
    gpu_scaled_attack = attacker.attack(gpu_image)
    print("GPU 攻击完成。")
    output_tensor = gpu_scaled_attack.detach().cpu()
    output_tensor.squeeze_(0)  # 移除批次维度
    plt.imshow(output_tensor.permute(1, 2, 0).numpy())  # 将张量从(C, H, W)转换为(H, W, C)
    plt.axis('off')  # 不显示坐标轴
    plt.tight_layout()
    plt.show()
    plt.close()  # 关闭图形，避免在内存中累积
    # 验证输出仍在 GPU 上
    print(f"输出张量设备: {gpu_scaled_attack.device}")

else:
    print("CUDA 不可用，请确保环境已配置。")