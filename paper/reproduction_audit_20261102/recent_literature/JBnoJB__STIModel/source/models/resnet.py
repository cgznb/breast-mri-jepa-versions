"""
ResNet model implementation for 3D medical images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial


def conv3x3x3(in_planes, out_planes, stride=1):
    """
    3x3x3 convolution with padding.
    
    Args:
        in_planes (int): Number of input channels
        out_planes (int): Number of output channels
        stride (int): Stride for the convolution
        
    Returns:
        nn.Conv3d: 3D convolutional layer
    """
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    """
    Basic downsampling block for residual connections.
    
    Args:
        x (torch.Tensor): Input tensor
        planes (int): Number of output channels
        stride (int): Stride for downsampling
        
    Returns:
        torch.Tensor: Downsampled tensor
    """
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.zeros(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4))
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = torch.cat([out.data, zero_pads], dim=1)

    return out


class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet.
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        Initialize the basic block.
        
        Args:
            inplanes (int): Number of input channels
            planes (int): Number of output channels
            stride (int): Stride for the first convolution
            downsample (nn.Module): Downsampling layer for residual connection
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        Forward pass of the basic block.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    Bottleneck residual block for ResNet.
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        Initialize the bottleneck block.
        
        Args:
            inplanes (int): Number of input channels
            planes (int): Number of intermediate channels
            stride (int): Stride for the first convolution
            downsample (nn.Module): Downsampling layer for residual connection
        """
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        Forward pass of the bottleneck block.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet model for 3D medical images.
    """
    def __init__(self,
                 block,
                 layers,
                 sample_input_D,
                 sample_input_H,
                 sample_input_W,
                 num_seg_classes,
                 shortcut_type='B',
                 no_cuda=False):
        """
        Initialize the ResNet model.
        
        Args:
            block: Type of residual block to use (BasicBlock or Bottleneck)
            layers (list): Number of blocks in each layer
            sample_input_D (int): Depth of input volume
            sample_input_H (int): Height of input volume
            sample_input_W (int): Width of input volume
            num_seg_classes (int): Number of segmentation classes
            shortcut_type (str): Type of shortcut connection ('A' or 'B')
            no_cuda (bool): Whether to disable CUDA
        """
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.no_cuda = no_cuda
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_seg_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        """
        Make a layer of residual blocks.
        
        Args:
            block: Type of residual block to use
            planes (int): Number of output channels
            blocks (int): Number of blocks in the layer
            shortcut_type (str): Type of shortcut connection
            stride (int): Stride for the first block
            
        Returns:
            nn.Sequential: Layer of residual blocks
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the ResNet model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, 1, D, H, W]
            
        Returns:
            torch.Tensor: Output tensor of shape [B, num_seg_classes]
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet10(**kwargs):
    """
    Construct a ResNet-10 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """
    Construct a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """
    Construct a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """
    Construct a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model 