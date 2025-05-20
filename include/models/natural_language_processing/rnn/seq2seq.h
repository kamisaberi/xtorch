#pragma once

#include "models/common.h"

namespace xt::models
{
    struct Seq2Seq : xt::Cloneable<Seq2Seq>
    {
    private:

    public:
        Seq2Seq(int num_classes /* classes */, int in_channels = 3/* input channels */);

        Seq2Seq(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };

}