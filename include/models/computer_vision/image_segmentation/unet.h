#pragma once

#include "../../common.h"


// #include "../../../exceptions/implementation.h"

namespace xt::models {
    // Double Convolution Block
    struct DoubleConv : xt::Module {
        DoubleConv(int in_channels, int out_channels);
        torch::Tensor forward(torch::Tensor x);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;
        torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
        torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
    };

//    TORCH_MODULE(DoubleConv);

    // U-Net Model
    struct UNetImpl : torch::nn::Module {
        UNetImpl(int in_channels, int out_channels);

        torch::Tensor forward(torch::Tensor x);

        DoubleConv enc1{nullptr}, enc2{nullptr}, enc3{nullptr}, bottleneck{nullptr};
        DoubleConv dec1{nullptr}, dec2{nullptr}, dec3{nullptr};
        torch::nn::ConvTranspose2d upconv1{nullptr}, upconv2{nullptr}, upconv3{nullptr};
        torch::nn::Conv2d out_conv{nullptr};
    };

    TORCH_MODULE(UNet);

    // Dice Loss for Binary Segmentation
    struct DiceLossImpl : torch::nn::Module {
        DiceLossImpl(float smooth = 1.0);

        torch::Tensor forward(torch::Tensor input, torch::Tensor target);

        float smooth_;
    };

    TORCH_MODULE(DiceLoss);
}
