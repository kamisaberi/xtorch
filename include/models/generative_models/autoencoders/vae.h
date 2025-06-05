#pragma once

#include "../../common.h"


namespace xt::models
{
    struct VAE : xt::Cloneable<VAE>
    {
    private:

    public:
        VAE(int num_classes /* classes */, int in_channels = 3/* input channels */);

        VAE(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };

}