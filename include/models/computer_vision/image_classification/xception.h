#pragma once

#include "../../common.h"

namespace xt::models
{
    // --- The Core Building Block: Depthwise Separable Convolution ---
    // struct SeparableConv2d : xt::Module
    // {
    //     torch::nn::Conv2d depthwise, pointwise;
    //
    //     SeparableConv2d(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0);
    //     auto forward(std::initializer_list<std::any> tensors) -> std::any override;
    //
    //     torch::Tensor forward(torch::Tensor x);
    // };
    //
    // // TORCH_MODULE(SeparableConv2d);
    //
    //
    // // --- The Main Repeating Block in Xception ---
    // struct XceptionBlock : xt::Module
    // {
    //     torch::nn::Sequential block, shortcut;
    //
    //     XceptionBlock(int in_channels, int out_channels, int num_reps, int stride, bool start_with_relu = true);
    //     auto forward(std::initializer_list<std::any> tensors) -> std::any override;
    //
    //     torch::Tensor forward(torch::Tensor x);
    // };
    //
    // // TORCH_MODULE(XceptionBlock);
    //
    //
    // // --- The Full Xception Model ---
    // struct Xception : xt::Module
    // {
    //     torch::nn::Conv2d conv1, conv2;
    //     torch::nn::BatchNorm2d bn1, bn2;
    //     torch::nn::Sequential entry_flow, middle_flow, exit_flow;
    //     torch::nn::Linear fc;
    //
    //     Xception(int num_middle_blocks, int num_classes = 10);
    //     auto forward(std::initializer_list<std::any> tensors) -> std::any override;
    //
    //     torch::Tensor forward(torch::Tensor x);
    // };

    // TORCH_MODULE(Xception);

    //    struct Xception : xt::Cloneable<Xception>
    //    {
    //    private:
    //
    //    public:
    //        Xception(int num_classes /* classes */, int in_channels = 3/* input channels */);
    //
    //        Xception(int num_classes, int in_channels, std::vector<int64_t> input_shape);
    //
    //        auto forward(std::initializer_list<std::any> tensors) -> std::any override;
    //        void reset() override;
    //    };
}
