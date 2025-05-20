#pragma once
#include "models/common.h"

namespace xt::models
{
    struct BART : xt::Cloneable<BART>
    {
    private:

    public:
        BART(int num_classes /* classes */, int in_channels = 3/* input channels */);

        BART(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };

}