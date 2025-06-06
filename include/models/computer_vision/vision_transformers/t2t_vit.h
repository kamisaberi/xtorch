#pragma once
#include "../../common.h"

namespace xt::models
{
    struct T2TViT : xt::Cloneable<T2TViT>
    {
    private:

    public:
        T2TViT(int num_classes /* classes */, int in_channels = 3/* input channels */);

        T2TViT(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;
        void reset() override;
    };

}