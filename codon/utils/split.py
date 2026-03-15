import torch.nn.functional as F
import torch
import numpy as np
import os
import base64

from PIL import Image
from io  import BytesIO

from typing import Union
from dataclasses import dataclass


@dataclass
class SplitedImage:
    '''
    Dataclass to hold the splited image patches and metadata.

    Attributes:
        patches (torch.Tensor): The flattened sequence of image patches.
            Shape: (num_patches, channels, patch_size, patch_size)
        original_shape (tuple): The original shape of the image (channels, height, width).
        grid_shape (tuple): The grid shape of the patches (num_patches_h, num_patches_w).
        patch_size (int): The size of each square patch.
        padded (bool): Whether padding was applied during splitting.
    '''
    patches: torch.Tensor
    original_shape: tuple
    grid_shape: tuple
    patch_size: int
    padded: bool


def split_image(
    image: Union[torch.Tensor, Image.Image, np.ndarray, str],
    patch_size: int = 8,
    padding: bool = False
) -> SplitedImage:
    '''
    Splits an image into a flattened 1D sequence of non-overlapping patches.

    Args:
        image (Union[torch.Tensor, Image.Image, np.ndarray, str]): The input image.
            Can be a file path, base64 string, PIL Image, numpy array, or PyTorch Tensor.
        patch_size (int): The size of each square patch. Defaults to 8.
        padding (bool): Whether to pad the image if its dimensions are not divisible
            by `patch_size`. If False, the image will be cropped. Defaults to False.

    Returns:
        SplitedImage: An object containing the patches and metadata.
    '''
    # Handle string inputs (file path or base64)
    if isinstance(image, str):
        if os.path.isfile(image):
            image = Image.open(image).convert('RGB')
        elif 'base64' in image or image.startswith('data:image'):
            if ',' in image:
                image_data = image.split(',')[1]
            else:
                image_data = image
            decoded = base64.b64decode(image_data)
            image = Image.open(BytesIO(decoded)).convert('RGB')
        else:
            raise ValueError('String input must be a valid file path or base64 string.')

    # Convert PIL Image to Tensor
    if isinstance(image, Image.Image):
        image = image.convert('RGB')
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

    # Convert numpy array to Tensor
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
        if image.shape[-1] in (1, 3, 4):
            # Assume HWC
            image_tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float()
        else:
            # Assume CHW
            image_tensor = torch.from_numpy(image.copy()).float()
            
        if image_tensor.max() > 1.0:
            image_tensor /= 255.0
        image = image_tensor

    # Validate Tensor input
    if not isinstance(image, torch.Tensor):
        raise TypeError(f'Unsupported image type: {type(image)}')

    # Ensure shape is (C, H, W)
    if image.ndim == 2:
        image = image.unsqueeze(0)
    elif image.ndim == 4:
        if image.shape[0] == 1:
            image = image.squeeze(0)
        else:
            raise ValueError('Expected single image tensor, got batch.')

    channels, height, width = image.shape
    original_shape = (channels, height, width)

    pad_h = (patch_size - height % patch_size) % patch_size
    pad_w = (patch_size - width % patch_size) % patch_size

    padded = False

    if pad_h > 0 or pad_w > 0:
        if padding:
            # F.pad format: (pad_left, pad_right, pad_top, pad_bottom)
            image = F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0.0)
            padded = True
        else:
            crop_h = height - (height % patch_size)
            crop_w = width - (width % patch_size)
            image = image[:, :crop_h, :crop_w]

    _, new_height, new_width = image.shape
    num_patches_h = new_height // patch_size
    num_patches_w = new_width // patch_size

    # image shape: (channels, new_height, new_width)
    patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    
    # Rearrange to sequence: (num_patches_h * num_patches_w, channels, patch_size, patch_size)
    patches = patches.permute(1, 2, 0, 3, 4).reshape(-1, channels, patch_size, patch_size)

    return SplitedImage(
        patches=patches,
        original_shape=original_shape,
        grid_shape=(num_patches_h, num_patches_w),
        patch_size=patch_size,
        padded=padded
    )
