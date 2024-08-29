from base import BaseModel

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import Tensor
from torch.distributions.normal import Normal

from typing import Optional, Tuple


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
class Standardize(nn.Module):
    def __init__(self):
        super(Standardize, self).__init__()
    
    def forward(self, x):
        mean = x.mean(dim=[0, 2, 3], keepdim=True)
        std = x.std(dim=[0, 2, 3], keepdim=True)
        return (x - mean) / std
class ResidualConvBlock(nn.Module):
    """Implements residual conv function.

    Args:
        channels (int): Number of channels in the input image.
    """

    def __init__(self, channels: int) -> None:
        super(ResidualConvBlock, self).__init__()
        self.rcb = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.rcb(x)
        out = torch.add(out, identity)

        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // 16, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)



class _InitialBlock(nn.Module):
    """Implementation of the input block of the ENet model.

    The initial block is composed of two branches:
        1. a main branch which performs a regular convolution with stride 2.
        2. an extension branch which performs max-pooling. Doing both operations in parallel and concatenating their
           results allows for efficient downsampling and expansion. The main branch outputs 13 feature maps while the
           extension branch outputs 3, for a total of 16 feature maps after concatenation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 0,
        bias: bool = False,
        relu: bool = True,
    ):
        """Initializes class instance.

        Args:
            in_channels: Number of input channels.
            out_channels: Number output channels.
            kernel_size: Kernel size of the filters used in the convolution layer.
            padding: Zero-padding added to both sides of the input.
            bias: Adds a learnable bias to the output if ``True``.
            relu: When ``True``, ReLU is used as the activation function; otherwise, PReLU is used.
        """
        super().__init__()

        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        # Main branch - As stated above the number of output channels for this
        # branch is the total minus 3, since the remaining channels come from
        # the extension branch
        self.main_branch = nn.Conv2d(
            in_channels,
            out_channels - in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
            bias=bias,
        )

        # Extension branch
        self.ext_branch = nn.MaxPool2d(kernel_size, stride=2, padding=padding)

        # Initialize batch normalization to be used after concatenation
        self.batch_norm = nn.BatchNorm2d(out_channels)

        # PReLU layer to apply after concatenating the branches
        self.out_activation = activation

    def forward(self, x: Tensor) -> Tensor:
        main = self.main_branch(x)
        ext = self.ext_branch(x)

        # Concatenate branches
        out = torch.cat((main, ext), 1)

        # Apply batch normalization
        out = self.batch_norm(out)

        return self.out_activation(out)


class _RegularBottleneck(nn.Module):
    """Regular bottlenecks are the main building block of ENet.

    Main branch:
        1. Shortcut connection.
    Extension branch:
        1. 1x1 convolution which decreases the number of channels by ``internal_ratio``, also called a projection;
        2. regular, dilated or asymmetric convolution;
        3. 1x1 convolution which increases the number of channels back to ``channels``, also called an expansion;
        4. dropout as a regularizer.
    """

    def __init__(
        self,
        channels: int,
        internal_ratio: int = 4,
        kernel_size: int = 3,
        padding: int = 0,
        dilation: int = 1,
        asymmetric: bool = False,
        dropout: float = 0.0,
        bias: bool = False,
        relu: bool = True,
    ):
        """Initializes class instance.

        Args:
            channels: Number of input and output channels.
            internal_ratio: Scale factor applied to ``channels`` used to compute the number of channels after the
                projection.

                e.g. given ``channels`` equal to 128 and internal_ratio equal to 2 the number of channels
                after the projection is 64.
            kernel_size: Kernel size of the filters used in the convolution layer described above in item 2 of the
                extension branch.
            padding: Zero-padding added to both sides of the input.
            dilation: Spacing between kernel elements for the convolution described in item 2 of the extension branch.
            asymmetric: Flags if the convolution described in item 2 of the extension branch is asymmetric or not.
            dropout: Probability of an element to be zeroed (e.g. 0 means no dropout).
            bias: Adds a learnable bias to the output if ``True``.
            relu: When ``True``, ReLU is used as the activation function; otherwise, PReLU is used.
        """
        super().__init__()

        # Check in the internal_ratio parameter is within the expected range [1, channels]
        if internal_ratio <= 1 or internal_ratio > channels:
            raise RuntimeError(
                f"Value out of range. Expected value in the interval [1, {channels}], "
                f"got internal_ratio={internal_ratio}."
            )

        internal_channels = channels // internal_ratio

        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        # Main branch - shortcut connection

        # Extension branch - 1x1 convolution, followed by a regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution, and,
        # finally, a regularizer (spatial dropout).
        # Number of channels is constant.

        # 1x1 projection convolution
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(channels, internal_channels, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(internal_channels),
            activation,
        )

        # If the convolution is asymmetric we split the main convolution in
        # two. Eg. for a 5x5 asymmetric convolution we have two convolution:
        # the first is 5x1 and the second is 1x5.
        if asymmetric:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(kernel_size, 1),
                    stride=1,
                    padding=(padding, 0),
                    dilation=dilation,
                    bias=bias,
                ),
                nn.BatchNorm2d(internal_channels),
                activation,
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(1, kernel_size),
                    stride=1,
                    padding=(0, padding),
                    dilation=dilation,
                    bias=bias,
                ),
                nn.BatchNorm2d(internal_channels),
                activation,
            )

        else:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=bias,
                ),
                nn.BatchNorm2d(internal_channels),
                activation,
            )

        # 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(internal_channels, channels, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(channels),
            activation,
        )

        self.ext_regul = nn.Dropout2d(p=dropout)

        # PReLU layer to apply after adding the branches
        self.out_prelu = activation

    def forward(self, x: Tensor) -> Tensor:
        # Main branch shortcut
        main = x

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Add main and extension branches
        out = main + ext

        return self.out_prelu(out)


class _DownsamplingBottleneck(nn.Module):
    """Downsampling bottlenecks further downsample the feature map size.

    Main branch:
        1. max pooling with stride 2; indices are saved to be used for unpooling later.
    Extension branch:
        1. 2x2 convolution with stride 2 that decreases the number of channels by ``internal_ratio``, also called a
           projection;
        2. regular convolution (by default, 3x3);
        3. 1x1 convolution which increases the number of channels to ``out_channels``, also called an expansion;
        4. dropout as a regularizer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        internal_ratio: int = 4,
        kernel_size: int = 3,
        padding: int = 0,
        return_indices: bool = False,
        dropout: float = 0.0,
        bias: bool = False,
        relu: bool = True,
    ):
        """Initializes class instance.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            internal_ratio: Scale factor applied to ``channels`` used to compute the number of channels after the
                projection.

                e.g. given ``channels`` equal to 128 and internal_ratio equal to 2 the number of channels after the
                projection is 64.
            kernel_size: Kernel size of the filters used in the convolution layer described above in item 2 of the
                extension branch.
            padding: Zero-padding added to both sides of the input.
            return_indices: If ``True``, will return the max indices along with the outputs. Useful when unpooling
                later.
            dropout: Probability of an element to be zeroed (e.g. 0 means no dropout).
            bias: Adds a learnable bias to the output if ``True``.
            relu: When ``True``, ReLU is used as the activation function; otherwise, PReLU is used.
        """
        super().__init__()

        # Store parameters that are needed later
        self.return_indices = return_indices

        # Check in the internal_ratio parameter is within the expected range [1, channels]
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError(
                f"Value out of range. Expected value in the interval [1, {in_channels}], "
                f"got internal_ratio={internal_ratio}."
            )

        internal_channels = in_channels // internal_ratio

        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        # Main branch - max pooling followed by feature map (channels) padding
        self.main_max1 = nn.MaxPool2d(
            kernel_size, stride=2, padding=padding, return_indices=return_indices
        )

        # Extension branch - 2x2 convolution, followed by a regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution. Number
        # of channels is doubled.

        # 2x2 projection convolution with stride 2
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, internal_channels, kernel_size=2, stride=2, bias=bias
            ),
            nn.BatchNorm2d(internal_channels),
            activation,
        )

        # Convolution
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                internal_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=bias,
            ),
            nn.BatchNorm2d(internal_channels),
            activation,
        )

        # 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(
                internal_channels, out_channels, kernel_size=1, stride=1, bias=bias
            ),
            nn.BatchNorm2d(out_channels),
            activation,
        )

        self.ext_regul = nn.Dropout2d(p=dropout)

        # PReLU layer to apply after concatenating the branches
        self.out_prelu = activation

    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        # Main branch shortcut
        if self.return_indices:
            main, max_indices = self.main_max1(x)
        else:
            main = self.main_max1(x)

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Main branch channel padding
        n, ch_ext, h, w = ext.size()
        ch_main = main.size()[1]
        padding = torch.zeros(n, ch_ext - ch_main, h, w)

        # Before concatenating, check if main is on the CPU or GPU and
        # convert padding accordingly
        if main.is_cuda:
            padding = padding.cuda()

        # Concatenate
        main = torch.cat((main, padding), 1)

        # Add main and extension branches
        out = main + ext

        return self.out_prelu(out), max_indices


class _UpsamplingBottleneck(nn.Module):
    """Upsampling bottleneck using max pooling indices stored from corresponding downsampling bottlenecks.

    Main branch:
        1. 1x1 convolution with stride 1 that decreases the number of channels by ``internal_ratio``, also called a
           projection;
        2. max unpool layer using the max pool indices from the corresponding downsampling max pool layer.
    Extension branch:
        1. 1x1 convolution with stride 1 that decreases the number of channels by ``internal_ratio``, also called a
           projection;
        2. transposed convolution (by default, 3x3);
        3. 1x1 convolution which increases the number of channels to ``out_channels``, also called an expansion;
        4. dropout as a regularizer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        internal_ratio: int = 4,
        kernel_size: int = 3,
        padding: int = 0,
        dropout: float = 0.0,
        bias: bool = False,
        relu: bool = True,
    ):
        """Initializes class instance.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            internal_ratio: Scale factor applied to ``in_channels`` used to compute the number of channels after the
                projection.

                e.g. given ``in_channels`` equal to 128 and ``internal_ratio`` equal to 2 the number of channels after
                the projection is 64.
            kernel_size: Kernel size of the filters used in the convolution layer described above in item 2 of the
                extension branch.
            padding: Zero-padding added to both sides of the input.
            dropout: Probability of an element to be zeroed (e.g. 0 means no dropout).
            bias: Adds a learnable bias to the output if ``True``.
            relu: When ``True``, ReLU is used as the activation function; otherwise, PReLU is used.
        """
        super().__init__()

        # Check in the internal_ratio parameter is within the expected range [1, channels]
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError(
                f"Value out of range. Expected value in the interval [1, {in_channels}], "
                f"got internal_ratio={internal_ratio}."
            )

        internal_channels = in_channels // internal_ratio

        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        # Main branch - max pooling followed by feature map (channels) padding
        self.main_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels),
        )

        # Remember that the stride is the same as the kernel_size, just like
        # the max pooling layers
        self.main_unpool1 = nn.MaxUnpool2d(kernel_size=2)

        # Extension branch - 1x1 convolution, followed by a regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution. Number
        # of channels is doubled.

        # 1x1 projection convolution with stride 1
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(internal_channels),
            activation,
        )

        # Transposed convolution
        self.ext_conv2 = nn.Sequential(
            nn.ConvTranspose2d(
                internal_channels,
                internal_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                output_padding=1,
                bias=bias,
            ),
            nn.BatchNorm2d(internal_channels),
            activation,
        )

        # 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            activation,
        )
        self.ext_regul = nn.Dropout2d(p=dropout)

        # PReLU layer to apply after concatenating the branches
        self.out_prelu = activation

    def forward(self, x: Tensor, max_indices) -> Tensor:
        # Main branch shortcut
        main = self.main_conv1(x)
        main = self.main_unpool1(main, max_indices)
        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Add main and extension branches
        out = main + ext

        return self.out_prelu(out)


class Enet(nn.Module):
    """Implementation of the ENet model.

    References:
        - Paper that introduced the model: http://arxiv.org/abs/1606.02147
    """

    def __init__(
        self,
        input_shape = [1,256,256],
        output_shape = [4,256,256],
        init_channels: int = 16,
        dropout: float = 0.1,
        encoder_relu: bool = True,
        decoder_relu: bool = True,
    ):
        """Initializes class instance.

        Args:
            input_shape: (in_channels, H, W), Shape of the input images.
            output_shape: (num_classes, H, W), Shape of the output segmentation map.
            init_channels: Number of output feature maps from the first layer, used to compute the number of feature
                maps in following layers.
            dropout: Probability of an element to be zeroed (e.g. 0 means no dropout).
                NOTE: In the initial block, the dropout probability is divided by 10.
            encoder_relu: When ``True`` ReLU is used as the activation function in the encoder blocks/layers; otherwise,
                PReLU is used.
            decoder_relu: When ``True`` ReLU is used as the activation function in the decoder blocks/layers; otherwise,
                PReLU is used.
        """
        super().__init__()
        in_channels = input_shape[0]
        out_channels = output_shape[0]
        self.dropout = dropout

        self.initial_block = _InitialBlock(
            in_channels, init_channels, padding=1, relu=encoder_relu
        )

        # Stage 1 - Encoder
        self.downsample1_0 = _DownsamplingBottleneck(
            init_channels,
            init_channels * 2,
            padding=1,
            return_indices=True,
            dropout=0.1 * dropout,
            relu=encoder_relu,
        )
        self.regular1_1 = _RegularBottleneck(
            init_channels * 2, padding=1, dropout=0.1 * dropout, relu=encoder_relu
        )
        self.regular1_2 = _RegularBottleneck(
            init_channels * 2, padding=1, dropout=0.1 * dropout, relu=encoder_relu
        )
        self.regular1_3 = _RegularBottleneck(
            init_channels * 2, padding=1, dropout=0.1 * dropout, relu=encoder_relu
        )
        self.regular1_4 = _RegularBottleneck(
            init_channels * 2, padding=1, dropout=0.1 * dropout, relu=encoder_relu
        )

        # Stage 2 - Encoder
        self.downsample2_0 = _DownsamplingBottleneck(
            init_channels * 2,
            init_channels * 4,
            padding=1,
            return_indices=True,
            dropout=dropout,
            relu=encoder_relu,
        )
        self.regular2_1 = _RegularBottleneck(
            init_channels * 4, padding=1, dropout=dropout, relu=encoder_relu
        )
        self.dilated2_2 = _RegularBottleneck(
            init_channels * 4, dilation=2, padding=2, dropout=dropout, relu=encoder_relu
        )
        self.asymmetric2_3 = _RegularBottleneck(
            init_channels * 4,
            kernel_size=5,
            padding=2,
            asymmetric=True,
            dropout=dropout,
            relu=encoder_relu,
        )
        self.dilated2_4 = _RegularBottleneck(
            init_channels * 4, dilation=4, padding=4, dropout=dropout, relu=encoder_relu
        )
        self.regular2_5 = _RegularBottleneck(
            init_channels * 4, padding=1, dropout=dropout, relu=encoder_relu
        )
        self.dilated2_6 = _RegularBottleneck(
            init_channels * 4, dilation=8, padding=8, dropout=dropout, relu=encoder_relu
        )
        self.asymmetric2_7 = _RegularBottleneck(
            init_channels * 4,
            kernel_size=5,
            asymmetric=True,
            padding=2,
            dropout=dropout,
            relu=encoder_relu,
        )
        self.dilated2_8 = _RegularBottleneck(
            init_channels * 4,
            dilation=16,
            padding=16,
            dropout=dropout,
            relu=encoder_relu,
        )

        # Stage 3 - Encoder
        self.regular3_0 = _RegularBottleneck(
            init_channels * 4, padding=1, dropout=dropout, relu=encoder_relu
        )
        self.dilated3_1 = _RegularBottleneck(
            init_channels * 4, dilation=2, padding=2, dropout=dropout, relu=encoder_relu
        )
        self.asymmetric3_2 = _RegularBottleneck(
            init_channels * 4,
            kernel_size=5,
            padding=2,
            asymmetric=True,
            dropout=dropout,
            relu=encoder_relu,
        )
        self.dilated3_3 = _RegularBottleneck(
            init_channels * 4, dilation=4, padding=4, dropout=dropout, relu=encoder_relu
        )
        self.regular3_4 = _RegularBottleneck(
            init_channels * 4, padding=1, dropout=dropout, relu=encoder_relu
        )
        self.dilated3_5 = _RegularBottleneck(
            init_channels * 4, dilation=8, padding=8, dropout=dropout, relu=encoder_relu
        )
        self.asymmetric3_6 = _RegularBottleneck(
            init_channels * 4,
            kernel_size=5,
            asymmetric=True,
            padding=2,
            dropout=dropout,
            relu=encoder_relu,
        )
        self.dilated3_7 = _RegularBottleneck(
            init_channels * 4,
            dilation=16,
            padding=16,
            dropout=dropout,
            relu=encoder_relu,
        )

        # Stage 4 - Decoder
        self.upsample4_0 = _UpsamplingBottleneck(
            init_channels * 4,
            init_channels * 2,
            padding=1,
            dropout=dropout,
            relu=decoder_relu,
        )
        self.regular4_1 = _RegularBottleneck(
            init_channels * 2, padding=1, dropout=dropout, relu=decoder_relu
        )
        self.regular4_2 = _RegularBottleneck(
            init_channels * 2, padding=1, dropout=dropout, relu=decoder_relu
        )

        # Stage 5 - Decoder
        self.upsample5_0 = _UpsamplingBottleneck(
            init_channels * 2,
            init_channels,
            padding=1,
            dropout=dropout,
            relu=decoder_relu,
        )
        self.regular5_1 = _RegularBottleneck(
            init_channels, padding=1, dropout=dropout, relu=decoder_relu
        )

        self.transposed_conv = nn.ConvTranspose2d(
            init_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            x: (N, ``in_channels``, H, W), Input image to segment.

        Returns:
            (N, ``out_channels``, H, W), Raw, unnormalized scores for each class in the input's segmentation.
        """
        # Initial block
        x = self.initial_block(x)

        # Stage 1 - Encoder
        x, max_indices1_0 = self.downsample1_0(x)
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x = self.regular1_4(x)

        # Stage 2 - Encoder
        x, max_indices2_0 = self.downsample2_0(x)
        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetric2_3(x)
        x = self.dilated2_4(x)
        x = self.regular2_5(x)
        x = self.dilated2_6(x)
        x = self.asymmetric2_7(x)
        x = self.dilated2_8(x)

        # Stage 3 - Encoder
        x = self.regular3_0(x)
        x = self.dilated3_1(x)
        x = self.asymmetric3_2(x)
        x = self.dilated3_3(x)
        x = self.regular3_4(x)
        x = self.dilated3_5(x)
        x = self.asymmetric3_6(x)
        x = self.dilated3_7(x)

        # Decoder
        # Stage 4 - Decoder
        x = self.upsample4_0(x, max_indices2_0)
        x = self.regular4_1(x)
        x = self.regular4_2(x)

        # Stage 5 - Decoder
        x = self.upsample5_0(x, max_indices1_0)
        x = self.regular5_1(x)
        return self.transposed_conv(x)




class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
class Standardize(nn.Module):
    def __init__(self):
        super(Standardize, self).__init__()
    
    def forward(self, x):
        mean = x.mean(dim=[0, 2, 3], keepdim=True)
        std = x.std(dim=[0, 2, 3], keepdim=True)
        return (x - mean) / std
class ResidualConvBlock(nn.Module):
    """Implements residual conv function.

    Args:
        channels (int): Number of channels in the input image.
    """

    def __init__(self, channels: int) -> None:
        super(ResidualConvBlock, self).__init__()
        self.rcb = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.rcb(x)
        out = torch.add(out, identity)

        return out


class GaussCap(nn.Module):
    """
    This class is a family of BayesCap framework
    It outputs a sigma of gaussian distribution
    """

    def __init__(self,
        input_shape = [1,256,256],
        output_shape = [4,256,256],
        ):
        super().__init__()
        latent_dim = 64
        
        input_in_channels = input_shape[0]
        output_in_channels = output_shape[0]
        output_out_channels = output_shape[0]
        #         
        self.conv_block_y1 = nn.Sequential(
            nn.Conv2d(output_in_channels, latent_dim, kernel_size=9, stride=1, padding=4),
            nn.PReLU(),
        )

        # Features trunk blocks.
        
        trunk_y = []
        for _ in range(4):
            trunk_y.append(ResidualConvBlock(latent_dim))
        self.trunk_y = nn.Sequential(*trunk_y)

        # Second conv layer.
        self.conv_block_y2 = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(latent_dim),
        )
        self.ca_y = ChannelAttention(in_planes=latent_dim)
        self.sa_y = SpatialAttention()        
        #
        self.conv_block_x1 = nn.Sequential(
            nn.Conv2d(input_in_channels, latent_dim, kernel_size=9, stride=1, padding=4),
            nn.PReLU(),
        )               
        # Features trunk blocks.
        trunk_x = []
        for _ in range(4):
            trunk_x.append(ResidualConvBlock(latent_dim))
        self.trunk_x = nn.Sequential(*trunk_x)

        # Second conv layer.
        self.conv_block_x2 = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(latent_dim),
        )
        
        self.ca_x = ChannelAttention(in_planes=latent_dim)
        self.sa_x = SpatialAttention()
        # Output layer.
        self.conv_block3_mu = nn.Conv2d(
            latent_dim, out_channels=output_out_channels, kernel_size=9, stride=1, padding=4
        )

        ## one_over_sigma
        self.conv_gauss = nn.Sequential(
            nn.Conv2d(latent_dim, 1, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
        )
        self.standardize_out = Standardize()
        self._initialize_weights()
        
    def uncertainty_map(self, x: Tensor, yhat: Tensor):
        
        ytilde = self.forward(x,yhat)
        alpha = torch.exp(ytilde)
        sum = alpha.sum(dim=1)
        alpha_tilde = alpha/sum.unsqueeze(1)
        uncertainty_map = ((alpha_tilde*(1-alpha_tilde))/(sum+1).unsqueeze(1)).sum(dim=1)
        uncertainty_map = uncertainty_map #*50
        uncertainty_map[uncertainty_map>1.]=1.
        
        return uncertainty_map
    def latent_out(self, x: Tensor, yhat: Tensor):
        out_y1 = self.conv_block_y1(yhat)
        
        out_y = self.trunk_y(out_y1)
        out_y2 = self.conv_block_y2(out_y)
        
        out_y = out_y1 + out_y2
        
        out_y = self.ca_y(out_y) * out_y
        out_y = self.sa_y(out_y) * out_y
        
        out_x1 = self.conv_block_x1(x)
        
        out_x = self.trunk_x(out_x1)
        out_x2 = self.conv_block_x2(out_x)
        out_x = out_x1 + out_x2     
        
        out_x1 = self.ca_x(out_x1) * out_x1
        out_x1 = self.sa_x(out_x1) * out_x1 
                  
        out = out_y + out_x
        out_norm = self.standardize_out(out)
        return out_norm
    # Support torch.script function.
    def forward(self, x: Tensor, yhat: Tensor):
        out_y1 = self.conv_block_y1(yhat)
        
        out_y = self.trunk_y(out_y1)
        out_y2 = self.conv_block_y2(out_y)
        
        out_y = out_y1 + out_y2
        
        out_y = self.ca_y(out_y) * out_y
        out_y = self.sa_y(out_y) * out_y
        
        out_x1 = self.conv_block_x1(x)
        
        out_x = self.trunk_x(out_x1)
        out_x2 = self.conv_block_x2(out_x)
        out_x = out_x1 + out_x2     
        
        out_x1 = self.ca_x(out_x1) * out_x1
        out_x1 = self.sa_x(out_x1) * out_x1 
                  
        out= out_y + out_x
        out = self.standardize_out(out)
        
        
        ytilde = self.conv_block3_mu(out)
        yvar = self.conv_gauss(out)
        # print(f"yvar: {yvar.shape, yvar.min(), yvar.max()}")
        # assert False
        return ytilde

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // 16, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)



