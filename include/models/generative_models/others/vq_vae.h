#pragma once
#include "models/common.h"

namespace xt::models
{
    struct QAVAEV1 : xt::Cloneable<QAVAEV1>
    {
    private:

    public:
        QAVAEV1(int num_classes /* classes */, int in_channels = 3/* input channels */);

        QAVAEV1(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };

    struct QAVAEV2 : xt::Cloneable<QAVAEV2>
    {
    private:

    public:
        QAVAEV2(int num_classes /* classes */, int in_channels = 3/* input channels */);

        QAVAEV2(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };

}