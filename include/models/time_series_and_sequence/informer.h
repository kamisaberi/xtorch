#pragma once
#include "models/common.h"

namespace xt::models
{
    struct Informer : xt::Cloneable<Informer>
    {
    private:

    public:
        Informer(int num_classes /* classes */, int in_channels = 3/* input channels */);

        Informer(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };

}