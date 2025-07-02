#pragma once
#include "../../common.h"


namespace xt::models
{
    struct BERT : xt::Cloneable<BERT>
    {
    private:

    public:
        BERT(int num_classes /* classes */, int in_channels = 3/* input channels */);

        BERT(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;
        void reset() override;
    };

}