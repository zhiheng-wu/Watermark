# from .core import AttackerFactory, BaseAttacker
# import torch
# # 恒等变换，用于测试，实际使用时注释掉
# @AttackerFactory.register("identity")
# class IdentityAttacker(BaseAttacker):
#     def attack(self, image: torch.Tensor) -> torch.Tensor:
#         return image