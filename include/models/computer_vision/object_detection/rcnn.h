#pragma once
#include "../../common.h"

namespace xt::models
{
    struct RCNN : xt::Cloneable<RCNN>
    {
    private:

    public:
        RCNN(int num_classes /* classes */, int in_channels = 3/* input channels */);

        RCNN(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;
        void reset() override;
    };

}