import math
import torch
import torch.nn as nn
from typing import Tuple, Optional

from codon.base import BasicModel
from codon.block.conv import ConvBlock


class PixelShuffleUpSample(BasicModel):
    '''
    Pixel Shuffle Upsampling Module.

    Supports 1D, 2D, and 3D data. This module uses a convolution to increase
    the number of channels, followed by a reshaping operation (Pixel Shuffle)
    to move channel spatial information to the spatial dimensions, effectively
    upsampling the input tensor.

    Attributes:
        conv (ConvBlock): The convolution block that projects the input to the required number of channels.
        dim (int): The dimensionality of the input data (1, 2, or 3).
        upscale_factor (int): The factor by which to upsample the spatial dimensions.
        out_channels (int): The final number of output channels after upsampling.
    '''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upscale_factor: int,
        dim: int = 2,
        norm: Optional[str] = None,
        activation: str = 'relu',
        dropout: float = 0.0
    ) -> None:
        '''
        Initializes the PixelShuffleUpSample module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels after pixel shuffle.
            upscale_factor (int): Factor to increase spatial resolution by.
            dim (int, optional): Dimensionality of the data (1, 2, or 3). Defaults to 2.
            norm (Optional[str], optional): Normalization type. Defaults to None.
            activation (str, optional): Activation function type. Defaults to 'relu'.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
        '''
        super().__init__()
        
        if dim not in (1, 2, 3):
            raise ValueError(f'Unsupported dimension: {dim}. Must be 1, 2, or 3.')

        self.dim = dim
        self.upscale_factor = upscale_factor
        self.out_channels = out_channels

        intermediate_channels = out_channels * (upscale_factor ** dim)

        self.conv = ConvBlock(
            in_channels=in_channels,
            out_channels=intermediate_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dim=dim,
            norm=norm,
            activation=activation,
            dropout=dropout
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        '''
        Defines the computation performed at every call.

        Args:
            input_tensor (torch.Tensor): The input data with shape (batch_size, in_channels, *spatial_dims).

        Returns:
            torch.Tensor: The upsampled output data with shape (batch_size, out_channels, *upsampled_spatial_dims).
        '''
        hidden = self.conv(input_tensor)
        
        batch_size, _, *spatial_dims = hidden.shape
        r = self.upscale_factor
        c = self.out_channels
        
        if self.dim == 1:
            l = spatial_dims[0]
            hidden = hidden.view(batch_size, c, r, l)
            hidden = hidden.permute(0, 1, 3, 2).contiguous()
            output = hidden.view(batch_size, c, l * r)
        elif self.dim == 2:
            h, w = spatial_dims
            hidden = hidden.view(batch_size, c, r, r, h, w)
            hidden = hidden.permute(0, 1, 4, 2, 5, 3).contiguous()
            output = hidden.view(batch_size, c, h * r, w * r)
        elif self.dim == 3:
            d, h, w = spatial_dims
            hidden = hidden.view(batch_size, c, r, r, r, d, h, w)
            hidden = hidden.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
            output = hidden.view(batch_size, c, d * r, h * r, w * r)
        else:
            raise ValueError(f'Unsupported dimension: {self.dim}')
            
        return output

    @staticmethod
    def auto_build(
        input_shape: Tuple[int, ...],
        output_shape: Optional[Tuple[int, ...]] = None,
        upscale_factor: Optional[int] = None,
        norm: Optional[str] = None,
        activation: str = 'relu',
        dropout: float = 0.0,
        depth_level: int = 1
    ) -> nn.Module:
        '''
        Automatically builds a PixelShuffleUpSample module or a Sequential of modules based on shapes.

        If the desired output spatial dimension is not an exact multiple of the input spatial dimension
        by the upscale factor, an AdaptiveAvgPool layer is appended to match the output shape exactly.

        Args:
            input_shape (Tuple[int, ...]): Shape of the input data (without batch size).
            output_shape (Optional[Tuple[int, ...]], optional): Shape of the desired output data. Defaults to None.
            upscale_factor (Optional[int], optional): Factor to increase spatial resolution by. Defaults to None.
            norm (Optional[str], optional): Normalization type. Defaults to None.
            activation (str, optional): Activation function type. Defaults to 'relu'.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            depth_level (int, optional): Level of network depth multiplier. Defaults to 1.

        Returns:
            nn.Module: An initialized PixelShuffleUpSample module or an nn.Sequential.
        '''
        dim = len(input_shape) - 1
        in_channels = input_shape[0]

        if output_shape is not None:
            out_channels = output_shape[0]
            if upscale_factor is None:
                upscale_factor = max(1, math.ceil(output_shape[1] / input_shape[1]))
        else:
            out_channels = in_channels
            if upscale_factor is None:
                upscale_factor = 2  # Default to 2x upsampling if neither output_shape nor factor is provided

        layers = []
        # Add upsampling block
        block = PixelShuffleUpSample(
            in_channels=in_channels,
            out_channels=out_channels,
            upscale_factor=upscale_factor,
            dim=dim,
            norm=norm,
            activation=activation,
            dropout=dropout
        )
        layers.append(block)

        # Add additional refinement layers based on depth_level
        for _ in range(max(0, depth_level - 1)):
            layers.append(ConvBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dim=dim,
                norm=norm,
                activation=activation,
                dropout=dropout
            ))

        if output_shape is not None:
            expected_spatial_size = input_shape[1] * upscale_factor
            target_spatial_size = output_shape[1]
            if expected_spatial_size != target_spatial_size:
                pool_layer = None
                target_spatial = output_shape[1:]
                if dim == 1:
                    pool_layer = nn.AdaptiveAvgPool1d(target_spatial)
                elif dim == 2:
                    pool_layer = nn.AdaptiveAvgPool2d(target_spatial)
                elif dim == 3:
                    pool_layer = nn.AdaptiveAvgPool3d(target_spatial)

                if pool_layer is not None:
                    layers.append(pool_layer)

        if len(layers) == 1:
            return layers[0]
        else:
            return nn.Sequential(*layers)
