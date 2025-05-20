#pragma once
#include "models/common.h"

namespace xt::models
{
    struct Word2Vec : xt::Cloneable<Word2Vec>
    {
    private:

    public:
        Word2Vec(int num_classes /* classes */, int in_channels = 3/* input channels */);

        Word2Vec(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };

}