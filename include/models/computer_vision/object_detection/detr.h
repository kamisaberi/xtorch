#pragma once
#include "../../common.h"



namespace xt::models
{
    struct DETR : xt::Cloneable<DETR>
    {
    private:

    public:
        DETR(int num_classes /* classes */, int in_channels = 3/* input channels */);

        DETR(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };

}