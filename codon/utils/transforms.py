import torch
from torchvision import transforms

class AddGaussianNoise(object):
    def __init__(
        self,
        mean: float = 0.0,
        std: float = 0.05,
        random: bool = False
    ) -> None:
        self.std = std
        self.mean = mean
        self.random = random
        
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        current_std = torch.rand(1).item() * self.std if self.random else self.std
        return tensor + torch.randn(tensor.size()) * current_std + self.mean

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std}, random={self.random})'

def _vision_transform(output_size=(240, 240)):
    return transforms.Compose([
        transforms.RandomResizedCrop(size=output_size, scale=(0.5, 1.2), ratio=(0.75, 1.33)),
        transforms.RandomRotation(degrees=30),
        transforms.ToTensor(),
        AddGaussianNoise(0.0, 0.05),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
