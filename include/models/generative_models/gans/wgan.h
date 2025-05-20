#pragma once
#include "models/common.h"

namespace xt::models
{
    struct WGAN : xt::Cloneable<WGAN>
    {
    private:

    public:
        WGAN(int num_classes /* classes */, int in_channels = 3/* input channels */);

        WGAN(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };

}