#pragma once
#include "models/common.h"
namespace xt::models
{
    struct PVTV1 : xt::Cloneable<PVTV1>
    {
    private:

    public:
        PVTV1(int num_classes /* classes */, int in_channels = 3/* input channels */);

        PVTV1(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };

    struct PVTV2 : xt::Cloneable<PVTV2>
    {
    private:

    public:
        PVTV2(int num_classes /* classes */, int in_channels = 3/* input channels */);

        PVTV2(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };

}