#pragma once

#include "../../common.h"


namespace xt::models
{
    // Depthwise Separable Convolution
    struct DepthwiseSeparableConv : xt::Module
    {
        DepthwiseSeparableConv(int in_channels, int out_channels, int stride);

        torch::Tensor forward(torch::Tensor x);

        torch::nn::Conv2d dw_conv{nullptr}, pw_conv{nullptr};
        torch::nn::BatchNorm2d dw_bn{nullptr}, pw_bn{nullptr};
    };

    // TORCH_MODULE(DepthwiseSeparableConv);

    // Sified MobileNetV1
    struct MobileNetV1 : xt::Module
    {
        MobileNetV1(int in_channels, int num_classes);

        torch::Tensor forward(torch::Tensor x);

        torch::nn::Conv2d stem_conv{nullptr};
        torch::nn::BatchNorm2d stem_bn{nullptr};
        torch::nn::Sequential blocks{nullptr};
        torch::nn::AdaptiveAvgPool2d pool{nullptr};
        torch::nn::Linear fc{nullptr};
    };

    // TORCH_MODULE(MobileNetV1);

    // Inverted Residual Block
    struct InvertedResidualBlock : xt::Module
    {
        InvertedResidualBlock(int in_channels, int exp_channels, int out_channels, int stride);

        torch::Tensor forward(torch::Tensor x);

        torch::nn::Conv2d expand_conv{nullptr}, dw_conv{nullptr}, project_conv{nullptr};
        torch::nn::BatchNorm2d expand_bn{nullptr}, dw_bn{nullptr}, project_bn{nullptr};
    };

    // TORCH_MODULE(InvertedResidualBlock);

    // Sified MobileNetV2
    struct MobileNetV2 : xt::Module
    {
        MobileNetV2(int in_channels, int num_classes);

        torch::Tensor forward(torch::Tensor x);

        torch::nn::Conv2d stem_conv{nullptr}, head_conv{nullptr};
        torch::nn::BatchNorm2d stem_bn{nullptr}, head_bn{nullptr};
        torch::nn::Sequential blocks{nullptr};
        torch::nn::AdaptiveAvgPool2d pool{nullptr};
        torch::nn::Linear fc{nullptr};
    };

    // TORCH_MODULE(MobileNetV2);


    // Squeeze-and-Excitation Module
    struct SEModule : xt::Module
    {
        SEModule(int in_channels, int reduction = 4);

        torch::Tensor forward(torch::Tensor x);

        torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    };


    // TORCH_MODULE(SEModule);

    // Sified MobileNetV3-Small
    struct MobileNetV3 : xt::Module
    {
        MobileNetV3(int in_channels, int num_classes);

        torch::Tensor forward(torch::Tensor x);

        torch::nn::Conv2d stem_conv{nullptr}, head_conv{nullptr};
        torch::nn::BatchNorm2d stem_bn{nullptr}, head_bn{nullptr};
        torch::nn::Sequential blocks{nullptr};
        torch::nn::AdaptiveAvgPool2d pool{nullptr};
        torch::nn::Linear fc{nullptr};
    };

    // TORCH_MODULE(MobileNetV3);
}
