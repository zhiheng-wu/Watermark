from generate import SD3BatchImageGenerator
import torch
# 使用示例
if __name__ == "__main__":
    def set_seed(seed: int = 42):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False
    
    set_seed(42)
    # 初始化生成器
    image_generator = SD3BatchImageGenerator(
        model_dir="D:/graduation/computer/Watermark/models/sd3.5",
        dataset_path="D:/graduation/computer/Watermark/dataset/prompts/stable_diffusion_prompts",
        max_images=20000
    )
    
    # 开始批量生成
    stats = image_generator.generate_batch(batch_size=5)
    
    # 如果有失败的任务，可以重试
    if stats["total_failed"] > 0:
        print(f"有 {stats['total_failed']} 个任务失败，开始重试...")
        image_generator.retry_failed()
    
    print("生成任务完成！")
    print(f"统计信息: {stats}")