#pragma once

#include "models/common.h"


using namespace std;


namespace xt::models
{
    struct EfficientNetB0 : xt::Cloneable<EfficientNetB0>
    {
    private:

    public:
        EfficientNetB0(int num_classes /* classes */, int in_channels = 3/* input channels */);

        EfficientNetB0(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };





}
