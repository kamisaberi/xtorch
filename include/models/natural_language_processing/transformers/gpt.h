#pragma once
#include "include/models/common.h"

namespace xt::models
{
    struct GPT : xt::Cloneable<GPT>
    {
    private:

    public:
        GPT(int num_classes /* classes */, int in_channels = 3/* input channels */);

        GPT(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };

}