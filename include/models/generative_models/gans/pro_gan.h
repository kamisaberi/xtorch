#pragma once
#include "../../common.h"


namespace xt::models
{
    struct ProGAN : xt::Cloneable<ProGAN>
    {
    private:

    public:
        ProGAN(int num_classes /* classes */, int in_channels = 3/* input channels */);

        ProGAN(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;
        void reset() override;
    };

}