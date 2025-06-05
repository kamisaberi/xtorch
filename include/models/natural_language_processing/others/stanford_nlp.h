#pragma once

#include "../../common.h"


namespace xt::models
{
    struct StanfordNLP : xt::Cloneable<StanfordNLP>
    {
    private:

    public:
        StanfordNLP(int num_classes /* classes */, int in_channels = 3/* input channels */);

        StanfordNLP(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };

}