#pragma once

#include "../../common.h"


using namespace std;

namespace xt::models
{
    // Operation: 3x3 Convolution
    struct Conv3x3Impl : torch::nn::Module
    {
        Conv3x3Impl(int in_channels, int out_channels)
        {
            conv = register_module("conv", torch::nn::Conv2d(
                                       torch::nn::Conv2dOptions(in_channels, out_channels, 3).stride(1).padding(1)));
            bn = register_module("bn", torch::nn::BatchNorm2d(out_channels));
        }

        torch::Tensor forward(torch::Tensor x)
        {
            return torch::relu(bn->forward(conv->forward(x)));
        }

        torch::nn::Conv2d conv{nullptr};
        torch::nn::BatchNorm2d bn{nullptr};
    };

    TORCH_MODULE(Conv3x3);

    // Operation: 1x1 Convolution
    struct Conv1x1Impl : torch::nn::Module
    {
        Conv1x1Impl(int in_channels, int out_channels)
        {
            conv = register_module("conv", torch::nn::Conv2d(
                                       torch::nn::Conv2dOptions(in_channels, out_channels, 1).stride(1)));
            bn = register_module("bn", torch::nn::BatchNorm2d(out_channels));
        }

        torch::Tensor forward(torch::Tensor x)
        {
            return torch::relu(bn->forward(conv->forward(x)));
        }

        torch::nn::Conv2d conv{nullptr};
        torch::nn::BatchNorm2d bn{nullptr};
    };

    TORCH_MODULE(Conv1x1);

    // Operation: 3x3 Max Pool
    struct MaxPool3x3Impl : torch::nn::Module
    {
        MaxPool3x3Impl()
        {
            pool = register_module("pool", torch::nn::MaxPool2d(
                                       torch::nn::MaxPool2dOptions(3).stride(1).padding(1)));
        }

        torch::Tensor forward(torch::Tensor x)
        {
            return pool->forward(x);
        }

        torch::nn::MaxPool2d pool{nullptr};
    };

    TORCH_MODULE(MaxPool3x3);

    // Normal Cell
    struct NormalCellImpl : torch::nn::Module
    {
        NormalCellImpl(int prev_channels, int channels)
        {
            // Simplified: Two branches (Conv3x3 + MaxPool3x3, Conv1x1)
            op1 = register_module("op1", Conv3x3(prev_channels, channels));
            op2 = register_module("op2", MaxPool3x3());
            op3 = register_module("op3", Conv1x1(prev_channels, channels));
        }

        torch::Tensor forward(torch::Tensor prev, torch::Tensor curr)
        {
            // Branch 1: Conv3x3(prev) + MaxPool3x3(curr)
            auto b1 = op1->forward(prev) + op2->forward(curr);
            // Branch 2: Conv1x1(curr)
            auto b2 = op3->forward(curr);
            // Combine
            return torch::cat({b1, b2}, 1); // [batch, 2*channels, h, w]
        }

        Conv3x3 op1{nullptr};
        MaxPool3x3 op2{nullptr};
        Conv1x1 op3{nullptr};
    };

    TORCH_MODULE(NormalCell);

    // Reduction Cell
    struct ReductionCellImpl : torch::nn::Module
    {
        ReductionCellImpl(int prev_channels, int channels)
        {
            // Simplified: Two branches (Conv3x3 stride 2, MaxPool3x3 stride 2)
            op1 = register_module("op1", torch::nn::Conv2d(
                                      torch::nn::Conv2dOptions(prev_channels, channels, 3).stride(2).padding(1)));
            bn1 = register_module("bn1", torch::nn::BatchNorm2d(channels));
            op2 = register_module("op2", torch::nn::MaxPool2d(
                                      torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));
            op3 = register_module("op3", Conv1x1(prev_channels, channels));
        }

        torch::Tensor forward(torch::Tensor prev, torch::Tensor curr)
        {
            // Branch 1: Conv3x3 stride 2(prev) + MaxPool3x3 stride 2(curr)
            auto b1 = torch::relu(bn1->forward(op1->forward(prev))) + op2->forward(curr);
            // Branch 2: Conv1x1(curr)
            auto b2 = op3->forward(curr);
            // Combine
            return torch::cat({b1, b2}, 1); // [batch, 2*channels, h/2, w/2]
        }

        torch::nn::Conv2d op1{nullptr};
        torch::nn::BatchNorm2d bn1{nullptr};
        torch::nn::MaxPool2d op2{nullptr};
        Conv1x1 op3{nullptr};
    };

    TORCH_MODULE(ReductionCell);

    // AmoebaNet-A (Simplified)
    struct AmoebaNetImpl : torch::nn::Module
    {
        AmoebaNetImpl(int in_channels, int num_classes, int channels = 64)
        {
            stem = register_module("stem", torch::nn::Conv2d(
                                       torch::nn::Conv2dOptions(in_channels, channels, 3).stride(1).padding(1)));
            bn_stem = register_module("bn_stem", torch::nn::BatchNorm2d(channels));
            normal_cell = register_module("normal_cell", NormalCell(channels, channels));
            reduction_cell = register_module("reduction_cell", ReductionCell(channels * 2, channels));
            classifier = register_module("classifier", torch::nn::Linear(4 * channels, num_classes));
            pool = register_module("pool", torch::nn::AdaptiveAvgPool2d(1));
        }

        torch::Tensor forward(torch::Tensor x)
        {
            // x: [batch, in_channels, 32, 32]
            auto h = torch::relu(bn_stem->forward(stem->forward(x))); // [batch, channels, 32, 32]
            auto prev = h;
            // Normal cell
            h = normal_cell->forward(prev, h); // [batch, 2*channels, 32, 32]
            prev = h;
            // Reduction cell
            h = reduction_cell->forward(prev, h); // [batch, 4*channels, 16, 16]
            // Global average pooling
            h = pool->forward(h); // [batch, 4*channels, 1, 1]
            h = h.view({h.size(0), -1}); // [batch, 4*channels]
            // Classifier
            return classifier->forward(h); // [batch, num_classes]
        }

        torch::nn::Conv2d stem{nullptr};
        torch::nn::BatchNorm2d bn_stem{nullptr};
        NormalCell normal_cell{nullptr};
        ReductionCell reduction_cell{nullptr};
        torch::nn::Linear classifier{nullptr};
        torch::nn::AdaptiveAvgPool2d pool{nullptr};
    };

    TORCH_MODULE(AmoebaNet);






    struct AmoabaNet : xt::Cloneable<AmoabaNet>
    {
    protected:

    public:
        explicit AmoabaNet(int num_classes/* classes */, int in_channels = 1/*  input channels */);
        AmoabaNet(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

        void reset() override;
    };
}
