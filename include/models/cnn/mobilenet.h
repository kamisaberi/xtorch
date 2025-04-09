#pragma once
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include "../base.h"


namespace xt::models {


//class SqueezeExcite(nn.Module):
//    def __init__(self,input_channels: int,squeeze: int = 4,) -> None:
//        """
//        Squeeze-and-Excitation block.
//
//        Args:
//        input_channels (`int`): Number of input channels.
//        squeeze (`int`, optional): Squeeze ratio. Defaults to 4.
//        """
//        super().__init__()
//
//        self.SE = nn.Sequential(
//                nn.AdaptiveAvgPool2d(output_size=1),
//                nn.Conv2d(input_channels, out_channels=input_channels//squeeze, kernel_size=1, stride=1, bias=False),
//        nn.BatchNorm2d(input_channels//squeeze),
//        nn.ReLU(inplace=True),
//        nn.Conv2d(input_channels//squeeze, input_channels, kernel_size=1, stride=1, bias=False),
//        nn.BatchNorm2d(input_channels),
//                HSigmoid(),
//        )
//
//    def forward(self, x):
//        x = x * self.SE(x)
//        return x



//class HSwish(nn.Module):
//    def __init__(self):
//        """Hard Swish activation function."""
//        super().__init__()
//        self.relu6 = nn.ReLU6(inplace=True)
//
//    def forward(self, x):
//        x = x * self.relu6(x + 3) / 6
//        return x
//
//
//class HSigmoid(nn.Module):
//    def __init__(self):
//        """Hard Sigmoid activation function."""
//        super().__init__()
//        self.relu6 = nn.ReLU6(inplace=True)
//
//    def forward(self, x):
//        x = self.relu6(x + 3) / 6
//        return x

//class Bottleneck(nn.Module):
//    def __init__(self,input_channels: int,kernel: int,stride: int,expansion: int,output_channels: int,activation: nn.Module,se: bool = False,) -> None:
//        """
//        MobileNetV3 bottleneck block.
//
//        Args:
//        input_channels (`int`): Number of input channels.
//        kernel (`int`): Convolution kernel size.
//        stride (`int`): Convolution stride.
//        expansion (`int`): Expansion size, indicating the middle layer's output channels.
//        output_channels (`int`): Number of final output channels.
//        activation (`nn.Module`): Activation function.
//        se (`bool`, optional): Whether to use Squeeze-and-Excitation. Defaults to False.
//        """
//        super().__init__()
//
//        self.bottleneck = nn.Sequential(
//        # expansion
//                nn.Conv2d(input_channels, expansion, kernel_size=1, stride=1, bias=False),
//                nn.BatchNorm2d(expansion),
//                activation,
//
//        # depth-wise convolution
//                nn.Conv2d(expansion, expansion, kernel_size=kernel, stride=stride, padding=kernel//2, groups=expansion, bias=False),
//        nn.BatchNorm2d(expansion),
//                activation,
//
//        # squeeze-and-excite
//                SqueezeExcite(expansion) if se else nn.Identity(),
//
//        # point-wise convolution
//                nn.Conv2d(expansion, output_channels, kernel_size=1, stride=1, bias=False),
//        nn.BatchNorm2d(output_channels),
//                activation,
//        )
//
//        # for residual skip connecting when the input size is different from output size
//        self.downsample = None if input_channels == output_channels and stride == 1 else nn.Sequential(
//                nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride, bias=False),
//        nn.BatchNorm2d(output_channels),
//        )
//
//
//    def forward(self, x):
//        residual = x
//        output = self.bottleneck(x)
//
//        if self.downsample:
//        residual = self.downsample(x)
//
//        output = output + residual
//
//        return output



//class MobileNetV3(nn.Module):
//    def __init__(self,input_channels: int,num_classes: int,dropout_prob: float = 0.5,) -> None:
//        """
//        MobileNetV3 (Large) model.
//
//        Args:
//        input_channels (`int`): Number of input channels.
//        num_classes (`int`): Number of classes.
//        dropout_prob (`float`, optional): Dropout probability. Defaults to 0.5.
//        """
//        super().__init__()
//
//        self.initial_conv = nn.Sequential(
//                nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=2),
//                nn.BatchNorm2d(16),
//                HSwish(),
//        )
//
//        self.bottlenecks = nn.Sequential(
//                Bottleneck(input_channels=16, kernel=3, stride=1, expansion=16, output_channels=16, activation=nn.ReLU(inplace=True)),
//                Bottleneck(input_channels=16, kernel=3, stride=2, expansion=64, output_channels=24, activation=nn.ReLU(inplace=True)),
//                Bottleneck(input_channels=24, kernel=3, stride=1, expansion=72, output_channels=24, activation=nn.ReLU(inplace=True)),
//                Bottleneck(input_channels=24, kernel=5, stride=2, expansion=72, output_channels=40, activation=nn.ReLU(inplace=True), se=True),
//                Bottleneck(input_channels=40, kernel=5, stride=1, expansion=120, output_channels=40, activation=nn.ReLU(inplace=True), se=True),
//                Bottleneck(input_channels=40, kernel=5, stride=1, expansion=120, output_channels=40, activation=nn.ReLU(inplace=True), se=True),
//                Bottleneck(input_channels=40, kernel=3, stride=2, expansion=240, output_channels=80, activation=HSwish()),
//                Bottleneck(input_channels=80, kernel=3, stride=1, expansion=200, output_channels=80, activation=HSwish()),
//                Bottleneck(input_channels=80, kernel=3, stride=1, expansion=184, output_channels=80, activation=HSwish()),
//                Bottleneck(input_channels=80, kernel=3, stride=1, expansion=184, output_channels=80, activation=HSwish()),
//                Bottleneck(input_channels=80, kernel=3, stride=1, expansion=480, output_channels=112, activation=HSwish(), se=True),
//                Bottleneck(input_channels=112, kernel=3, stride=1, expansion=672, output_channels=112, activation=HSwish(), se=True),
//                Bottleneck(input_channels=112, kernel=5, stride=2, expansion=672, output_channels=160, activation=HSwish(), se=True),
//                Bottleneck(input_channels=160, kernel=5, stride=1, expansion=960, output_channels=160, activation=HSwish(), se=True),
//                Bottleneck(input_channels=160, kernel=5, stride=1, expansion=960, output_channels=160, activation=HSwish(), se=True),
//        )
//
//        self.final_conv = nn.Sequential(
//                nn.Conv2d(in_channels=160, out_channels=960, kernel_size=1, stride=1, bias=False),
//                nn.BatchNorm2d(960),
//                HSwish(),
//        )
//
//        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
//
//        self.classifier = nn.Sequential(
//                nn.Linear(960, 1280),
//                HSwish(),
//                nn.Dropout(p=dropout_prob, inplace=True),
//                nn.Linear(1280, num_classes),
//        )
//
//    def forward(self, x):
//        x = self.initial_conv(x)
//        x = self.bottlenecks(x)
//        x = self.final_conv(x)
//        x = self.pool(x)
//        x = torch.flatten(x, 1)
//        x = self.classifier(x)
//        return x

    struct MobileNetV3 :BaseModel {
    public:
        MobileNetV3();


    };
}