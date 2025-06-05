#pragma once
#include "../../common.h"


namespace xt::models
{
    struct Flair : xt::Cloneable<Flair>
    {
    private:

    public:
        Flair(int num_classes /* classes */, int in_channels = 3/* input channels */);

        Flair(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };

}