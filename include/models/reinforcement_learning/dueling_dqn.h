#pragma once
#include "../common.h"

namespace xt::models
{
    struct DuelingDQN : xt::Cloneable<DuelingDQN>
    {
    private:

    public:
        DuelingDQN(int num_classes /* classes */, int in_channels = 3/* input channels */);

        DuelingDQN(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };

}