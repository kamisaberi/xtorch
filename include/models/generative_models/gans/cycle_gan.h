#pragma once

#include "../../common.h"


namespace xt::models
{
    struct CycleGAN : xt::Cloneable<CycleGAN>
    {
    private:

    public:
        struct ResidualBlock : torch::nn::Module
        {
        public:
            ResidualBlock(int64_t channels);
            torch::Tensor forward(torch::Tensor x);

        private:
            torch::nn::Conv2d conv1, conv2;
            torch::nn::InstanceNorm2d norm1, norm2;
        };

        CycleGAN(int num_classes /* classes */, int in_channels = 3/* input channels */);

        CycleGAN(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;
        void reset() override;
    };
}
