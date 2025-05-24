#pragma once
#include "include/models/common.h"
namespace xt::models
{
    struct CLIPViT : xt::Cloneable<CLIPViT>
    {
    private:

    public:
        CLIPViT(int num_classes /* classes */, int in_channels = 3/* input channels */);

        CLIPViT(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };

}