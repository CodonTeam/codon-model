import torch
from torchvision import transforms

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.05):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

def _vision_transform(output_size=(240, 240)):
    return transforms.Compose([
        transforms.RandomResizedCrop(size=output_size, scale=(0.5, 1.2), ratio=(0.75, 1.33)),
        transforms.RandomRotation(degrees=30),
        transforms.ToTensor(),
        AddGaussianNoise(0.0, 0.05),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])