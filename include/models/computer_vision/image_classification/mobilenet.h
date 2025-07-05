#pragma once

#include "../../common.h"


namespace xt::models
{
    // Squeeze-and-Excitation Module
    struct SEModuleImpl : torch::nn::Module
    {
        SEModuleImpl(int in_channels, int reduction = 4)
        {
            fc1 = register_module("fc1", torch::nn::Linear(in_channels, in_channels / reduction));
            fc2 = register_module("fc2", torch::nn::Linear(in_channels / reduction, in_channels));
        }

        torch::Tensor forward(torch::Tensor x)
        {
            // x: [batch, in_channels, h, w]
            auto avg_pool = torch::avg_pool2d(x, {x.size(2), x.size(3)}).squeeze(-1).squeeze(-1);
            // [batch, in_channels]
            auto se = torch::relu(fc1->forward(avg_pool)); // [batch, in_channels/reduction]
            se = torch::sigmoid(fc2->forward(se)).unsqueeze(-1).unsqueeze(-1); // [batch, in_channels, 1, 1]
            return x * se; // Element-wise multiplication
        }

        torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    };


    TORCH_MODULE(SEModule);

    // Depthwise Separable Convolution
    struct DepthwiseSeparableConvImpl : torch::nn::Module
    {
        DepthwiseSeparableConvImpl(int in_channels, int out_channels, int stride)
        {
            dw_conv = register_module("dw_conv", torch::nn::Conv2d(
                                          torch::nn::Conv2dOptions(in_channels, in_channels, 3).stride(stride).
                                          padding(1).groups(in_channels)));
            dw_bn = register_module("dw_bn", torch::nn::BatchNorm2d(in_channels));
            pw_conv = register_module("pw_conv", torch::nn::Conv2d(
                                          torch::nn::Conv2dOptions(in_channels, out_channels, 1).stride(1)));
            pw_bn = register_module("pw_bn", torch::nn::BatchNorm2d(out_channels));
        }

        torch::Tensor forward(torch::Tensor x)
        {
            x = torch::relu(dw_bn->forward(dw_conv->forward(x))); // Depthwise
            x = torch::relu(pw_bn->forward(pw_conv->forward(x))); // Pointwise
            return x;
        }

        torch::nn::Conv2d dw_conv{nullptr}, pw_conv{nullptr};
        torch::nn::BatchNorm2d dw_bn{nullptr}, pw_bn{nullptr};
    };

    TORCH_MODULE(DepthwiseSeparableConv);

    // Inverted Residual Block
    struct InvertedResidualBlockImpl : torch::nn::Module
    {
        InvertedResidualBlockImpl(int in_channels, int exp_channels, int out_channels, int stride, bool use_se)
            : use_se_(use_se)
        {
            expand_conv = nullptr;
            if (in_channels != exp_channels)
            {
                expand_conv = register_module("expand_conv", torch::nn::Conv2d(
                                                  torch::nn::Conv2dOptions(in_channels, exp_channels, 1).stride(1)));
                expand_bn = register_module("expand_bn", torch::nn::BatchNorm2d(exp_channels));
            }
            dw_conv = register_module("dw_conv", torch::nn::Conv2d(
                                          torch::nn::Conv2dOptions(exp_channels, exp_channels, 3).stride(stride).
                                          padding(1).groups(exp_channels)));
            dw_bn = register_module("dw_bn", torch::nn::BatchNorm2d(exp_channels));
            if (use_se)
            {
                se = register_module("se", SEModule(exp_channels));
            }
            project_conv = register_module("project_conv", torch::nn::Conv2d(
                                               torch::nn::Conv2dOptions(exp_channels, out_channels, 1).stride(1)));
            project_bn = register_module("project_bn", torch::nn::BatchNorm2d(out_channels));
        }

        torch::Tensor forward(torch::Tensor x)
        {
            auto residual = x;
            // Expansion
            if (expand_conv)
            {
                x = torch::relu(expand_bn->forward(expand_conv->forward(x)));
            }
            // Depthwise
            x = torch::relu(dw_bn->forward(dw_conv->forward(x)));
            // Squeeze-and-Excitation
            if (use_se_)
            {
                x = se->forward(x);
            }
            // Projection
            x = project_bn->forward(project_conv->forward(x));
            // Residual connection if shapes match
            if (x.sizes() == residual.sizes())
            {
                x = x + residual;
            }
            return x;
        }

        bool use_se_;
        torch::nn::Conv2d expand_conv{nullptr}, dw_conv{nullptr}, project_conv{nullptr};
        torch::nn::BatchNorm2d expand_bn{nullptr}, dw_bn{nullptr}, project_bn{nullptr};
        SEModule se{nullptr};
    };

    TORCH_MODULE(InvertedResidualBlock);

    // Simplified MobileNetV3-Small
    struct MobileNetV3Impl : torch::nn::Module
    {
        MobileNetV3Impl(int in_channels, int num_classes)
        {
            // Stem
            stem_conv = register_module("stem_conv", torch::nn::Conv2d(
                                            torch::nn::Conv2dOptions(in_channels, 16, 3).stride(2).padding(1)));
            stem_bn = register_module("stem_bn", torch::nn::BatchNorm2d(16));

            // Blocks: {in_channels, exp_channels, out_channels, stride, use_se}
            blocks = torch::nn::Sequential(
                InvertedResidualBlock(16, 16, 16, 2, true), // [batch, 16, 8, 8]
                InvertedResidualBlock(16, 72, 24, 2, false), // [batch, 24, 4, 4]
                InvertedResidualBlock(24, 88, 24, 1, false), // [batch, 24, 4, 4]
                InvertedResidualBlock(24, 96, 40, 2, true) // [batch, 40, 2, 2]
            );
            register_module("blocks", blocks);

            // Head
            head_conv = register_module("head_conv", torch::nn::Conv2d(
                                            torch::nn::Conv2dOptions(40, 128, 1).stride(1)));
            head_bn = register_module("head_bn", torch::nn::BatchNorm2d(128));
            pool = register_module("pool", torch::nn::AdaptiveAvgPool2d(1));
            fc = register_module("fc", torch::nn::Linear(128, num_classes));
        }

        torch::Tensor forward(torch::Tensor x)
        {
            // x: [batch, in_channels, 32, 32]
            x = torch::relu(stem_bn->forward(stem_conv->forward(x))); // [batch, 16, 16, 16]
            x = blocks->forward(x); // [batch, 40, 2, 2]
            x = torch::relu(head_bn->forward(head_conv->forward(x))); // [batch, 128, 2, 2]
            x = pool->forward(x).view({x.size(0), -1}); // [batch, 128]
            x = fc->forward(x); // [batch, num_classes]
            return x;
        }

        torch::nn::Conv2d stem_conv{nullptr}, head_conv{nullptr};
        torch::nn::BatchNorm2d stem_bn{nullptr}, head_bn{nullptr};
        torch::nn::Sequential blocks{nullptr};
        torch::nn::AdaptiveAvgPool2d pool{nullptr};
        torch::nn::Linear fc{nullptr};
    };

    TORCH_MODULE(MobileNetV3);
}
