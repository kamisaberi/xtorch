#pragma once
#include "../common.h"

namespace xt::models
{
    struct AlphaZero : xt::Cloneable<AlphaZero>
    {
    private:

    public:
        AlphaZero(int num_classes /* classes */, int in_channels = 3/* input channels */);

        AlphaZero(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;
        void reset() override;
    };

}