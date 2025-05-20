#pragma once
#include "models/common.h"
namespace xt::models
{

    struct DeepLabV1 : xt::Cloneable<DeepLabV1>
    {
    private:

    public:
        DeepLabV1(int num_classes /* classes */, int in_channels = 3/* input channels */);

        DeepLabV1(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };



}