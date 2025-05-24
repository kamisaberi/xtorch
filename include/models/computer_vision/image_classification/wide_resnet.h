#pragma once
#include "include/models/common.h"


using namespace std;

namespace xt::models {
    // namespace {
    //     struct ResidualBlock : torch::nn::Module {
    //         torch::nn::Sequential conv1 = nullptr, conv2 = nullptr, downsample = nullptr;
    //         int out_channels;
    //         torch::nn::ReLU relu = nullptr;
    //         torch::Tensor residual;

    //         ResidualBlock(int in_channels, int out_channels, int stride = 1,
    //                       torch::nn::Sequential downsample = nullptr);

    //         torch::Tensor forward(torch::Tensor x);
    //     };
    // }

    struct WideResNet : xt::Cloneable<WideResNet>
    {
    private:

    public:
        WideResNet(int num_classes /* classes */, int in_channels = 3/* input channels */);

        WideResNet(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };


}
