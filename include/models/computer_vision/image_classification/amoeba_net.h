#pragma once

#include "../../common.h"


using namespace std;

namespace xt::models {
    // Operation: 3x3 Convolution
    struct Conv3x3 : xt::Module {
        Conv3x3(int in_channels, int out_channels);

        auto forward(std::initializer_list <std::any> tensors) -> std::any override;
        torch::Tensor forward(torch::Tensor x);

        torch::nn::Conv2d conv{nullptr};
        torch::nn::BatchNorm2d bn{nullptr};
    };

//    TORCH_MODULE(Conv3x3);

    // Operation: 1x1 Convolution
    struct Conv1x1 : xt::Module {
        Conv1x1(int in_channels, int out_channels);

        auto forward(std::initializer_list <std::any> tensors) -> std::any override;
        torch::Tensor forward(torch::Tensor x);

        torch::nn::Conv2d conv{nullptr};
        torch::nn::BatchNorm2d bn{nullptr};
    };

//    TORCH_MODULE(Conv1x1);

    // Operation: 3x3 Max Pool
    struct MaxPool3x3Impl : torch::nn::Module {
        MaxPool3x3Impl();

        torch::Tensor forward(torch::Tensor x);

        torch::nn::MaxPool2d pool{nullptr};
    };

    TORCH_MODULE(MaxPool3x3);

    // Normal Cell
    struct NormalCellImpl : torch::nn::Module {
        NormalCellImpl(int prev_channels, int channels);

        torch::Tensor forward(torch::Tensor prev, torch::Tensor curr);

        Conv3x3 op1{nullptr};
        MaxPool3x3 op2{nullptr};
        Conv1x1 op3{nullptr};
    };

    TORCH_MODULE(NormalCell);

    // Reduction Cell
    struct ReductionCellImpl : torch::nn::Module {
        ReductionCellImpl(int prev_channels, int channels);

        torch::Tensor forward(torch::Tensor prev, torch::Tensor curr);

        torch::nn::Conv2d op1{nullptr};
        torch::nn::BatchNorm2d bn1{nullptr};
        torch::nn::MaxPool2d op2{nullptr};
        Conv1x1 op3{nullptr};
    };

    TORCH_MODULE(ReductionCell);

    // AmoebaNet-A (Simplified)
    struct AmoebaNetImpl : torch::nn::Module {
        AmoebaNetImpl(int in_channels, int num_classes, int channels = 64);

        torch::Tensor forward(torch::Tensor x);

        torch::nn::Conv2d stem{nullptr};
        torch::nn::BatchNorm2d bn_stem{nullptr};
        NormalCell normal_cell{nullptr};
        ReductionCell reduction_cell{nullptr};
        torch::nn::Linear classifier{nullptr};
        torch::nn::AdaptiveAvgPool2d pool{nullptr};
    };

    TORCH_MODULE(AmoebaNet);


    struct AmoabaNet : xt::Cloneable<AmoabaNet> {
    protected:

    public:
        explicit AmoabaNet(int num_classes/* classes */, int in_channels = 1/*  input channels */);

        AmoabaNet(int num_classes, int in_channels, std::vector <int64_t> input_shape);

        auto forward(std::initializer_list <std::any> tensors) -> std::any override;

        void reset() override;
    };
}
