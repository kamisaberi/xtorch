#pragma once
#include "../../common.h"


namespace xt::models
{
    struct Pix2Pix : xt::Cloneable<Pix2Pix>
    {
    private:

    public:
        Pix2Pix(int num_classes /* classes */, int in_channels = 3/* input channels */);

        Pix2Pix(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;
        void reset() override;
    };

}