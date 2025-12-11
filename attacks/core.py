from abc import ABC, abstractmethod
import torch


class BaseAttacker(ABC):
    def __init__(self, **kwargs):
        self.params = kwargs
    
    @abstractmethod
    def attack(self, image) -> torch.Tensor: 
        pass

class AttackerFactory:
    _registry = {}

    @classmethod
    def register(cls, name: str):
        def decorator(subclass):
            cls._registry[name] = subclass
            return subclass
        return decorator

    @classmethod
    def create_attacker(cls, name, params=None):
        if name not in cls._registry:
            raise ValueError(f"Attacker '{name}' not found. Registered: {list(cls._registry.keys())}")
        return cls._registry[name](**(params or {}))