#pragma once
#include "../../common.h"

namespace xt::models
{
    struct SwinTransformerV1 : xt::Cloneable<SwinTransformerV1>
    {
    private:

    public:
        SwinTransformerV1(int num_classes /* classes */, int in_channels = 3/* input channels */);

        SwinTransformerV1(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };

    struct SwinTransformerV2 : xt::Cloneable<SwinTransformerV2>
    {
    private:

    public:
        SwinTransformerV2(int num_classes /* classes */, int in_channels = 3/* input channels */);

        SwinTransformerV2(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };

}