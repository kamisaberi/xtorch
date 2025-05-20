#pragma once
#include "models/common.h"

namespace xt::models
{
    struct ProGAN : xt::Cloneable<ProGAN>
    {
    private:

    public:
        ProGAN(int num_classes /* classes */, int in_channels = 3/* input channels */);

        ProGAN(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };

}