#pragma once
#include "models/common.h"

namespace xt::models
{
    struct A3C  : xt::Cloneable<A3C >
    {
    private:

    public:
        A3C (int num_classes /* classes */, int in_channels = 3/* input channels */);

        A3C (int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };

}