#pragma once
#include "models/common.h"

namespace xt::models
{
    struct VilBERT : xt::Cloneable<VilBERT>
    {
    private:

    public:
        VilBERT(int num_classes /* classes */, int in_channels = 3/* input channels */);

        VilBERT(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };

}