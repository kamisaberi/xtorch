#pragma once
#include "../../common.h"



namespace xt::models
{
    struct FasterRCNN : xt::Cloneable<FasterRCNN>
    {
    private:

    public:
        FasterRCNN(int num_classes /* classes */, int in_channels = 3/* input channels */);

        FasterRCNN(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;
        void reset() override;
    };

}