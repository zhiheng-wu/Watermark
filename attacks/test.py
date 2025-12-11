from .core import AttackerFactory, BaseAttacker
import numpy as np
import torch
@AttackerFactory.register("identity")
class IdentityAttacker(BaseAttacker):
    def attack(self, image: torch.Tensor) -> torch.Tensor:
        return image