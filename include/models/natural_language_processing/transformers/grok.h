#pragma once
#include "../../common.h"


namespace xt::models
{
    struct Grok : xt::Cloneable<Grok>
    {
    private:

    public:
        Grok(int num_classes /* classes */, int in_channels = 3/* input channels */);

        Grok(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };

}