#pragma once
#include "../common.h"

namespace xt::models
{
    struct A3C  : xt::Cloneable<A3C >
    {
    private:

    public:
        A3C (int num_classes /* classes */, int in_channels = 3/* input channels */);

        A3C (int num_classes, int in_channels, std::vector<int64_t> input_shape);

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;
        void reset() override;
    };

}