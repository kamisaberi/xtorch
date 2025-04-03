#pragma once
#include <torch/torch.h>
#include "../../exceptions/implementation.h"

namespace xt::models {


//class DoubleConv(nn.Module):
//    def __init__(self, in_channels, out_channels):
//        super().__init__()
//        self.conv_op = nn.Sequential(
//                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
//                nn.ReLU(inplace=True),
//                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
//                nn.ReLU(inplace=True)
//        )
//
//    def forward(self, x):
//        return self.conv_op(x)


    struct DoubleConv : torch::nn::Module {
    public:


    };


//class DownSample(nn.Module):
//    def __init__(self, in_channels, out_channels):
//        super().__init__()
//        self.conv = DoubleConv(in_channels, out_channels)
//        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
//
//    def forward(self, x):
//        down = self.conv(x)
//        p = self.pool(down)
//        return down, p

    struct DownSample : torch::nn::Module {
    public:

    };



//class UpSample(nn.Module):
//    def __init__(self, in_channels, out_channels):
//        super().__init__()
//        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
//        self.conv = DoubleConv(in_channels, out_channels)
//
//    def forward(self, x1, x2):
//        x1 = self.up(x1)
//        x = torch.cat([x1, x2], 1)
//        return self.conv(x)

    struct UpSample : torch::nn::Module {
    public:

    };


//class UNet(nn.Module):
//    def __init__(self, in_channels, num_classes):
//        super().__init__()
//        self.down_convolution_1 = DownSample(in_channels, 64)
//        self.down_convolution_2 = DownSample(64, 128)
//        self.down_convolution_3 = DownSample(128, 256)
//        self.down_convolution_4 = DownSample(256, 512)
//
//        self.bottle_neck = DoubleConv(512, 1024)
//
//        self.up_convolution_1 = UpSample(1024, 512)
//        self.up_convolution_2 = UpSample(512, 256)
//        self.up_convolution_3 = UpSample(256, 128)
//        self.up_convolution_4 = UpSample(128, 64)
//
//        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)
//
//    def forward(self, x):
//        down_1, p1 = self.down_convolution_1(x)
//        down_2, p2 = self.down_convolution_2(p1)
//        down_3, p3 = self.down_convolution_3(p2)
//        down_4, p4 = self.down_convolution_4(p3)
//
//        b = self.bottle_neck(p4)
//
//        up_1 = self.up_convolution_1(b, down_4)
//        up_2 = self.up_convolution_2(up_1, down_3)
//        up_3 = self.up_convolution_3(up_2, down_2)
//        up_4 = self.up_convolution_4(up_3, down_1)
//
//        out = self.out(up_4)
//        return out
//


    struct UNet : torch::nn::Module {
    public:
        UNet();

    };
}