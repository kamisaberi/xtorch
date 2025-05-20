#pragma once
#include "models/common.h"
namespace xt::models
{
    struct DeiT : xt::Cloneable<DeiT>
    {
    private:

    public:
        DeiT(int num_classes /* classes */, int in_channels = 3/* input channels */);

        DeiT(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };

}