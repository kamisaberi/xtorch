#pragma once
#include "../../common.h"



namespace xt::models
{
    struct FastRCNN : xt::Cloneable<FastRCNN>
    {
    private:

    public:
        FastRCNN(int num_classes /* classes */, int in_channels = 3/* input channels */);

        FastRCNN(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;
        void reset() override;
    };

}