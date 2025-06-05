#pragma once
#include "../common.h"

namespace xt::models
{
    struct MERT : xt::Cloneable<MERT>
    {
    private:

    public:
        MERT(int num_classes /* classes */, int in_channels = 3/* input channels */);

        MERT(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };

}