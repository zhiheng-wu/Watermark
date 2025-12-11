import torch
from attacks.core import AttackerFactory
import matplotlib.pyplot as plt
import torchvision
import time


# 使用matplotlib的imshow显示和保存
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"CUDA可用，将在 {device} 上测试。")

    # 取一张测试图像 (C, H, W 格式，转换为浮点数，放在 GPU 上)
    gpu_image = torchvision.io.read_image("./dataset/origin/000000.png").float()/255.0
    gpu_image=gpu_image.to(device)
    print(f"输入张量设备: {gpu_image.device}")

    # 实例化 GPU 攻击者
    gpu_scaler = AttackerFactory.create_attacker("cropping", params={"crop_ratio":0.25,"crop_start_h":0,"crop_start_w":0})
    
    # 执行攻击
    gpu_scaled_attack = gpu_scaler.attack(gpu_image)
    output_tensor = gpu_scaled_attack.detach().cpu()
    plt.imshow(output_tensor.permute(1, 2, 0).numpy())  # 将张量从(C, H, W)转换为(H, W, C)
    plt.axis('off')  # 不显示坐标轴
    plt.tight_layout()
    plt.show()
    plt.close()  # 关闭图形，避免在内存中累积
    # 验证输出仍在 GPU 上
    print(f"输出张量设备: {gpu_scaled_attack.device}")

else:
    print("CUDA 不可用，请确保环境已配置。")