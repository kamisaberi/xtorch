#pragma once
#include "include/models/common.h"

namespace xt::models
{
    struct FastText : xt::Cloneable<FastText>
    {
    private:

    public:
        FastText(int num_classes /* classes */, int in_channels = 3/* input channels */);

        FastText(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };

}