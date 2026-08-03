#pragma once

#include "common.h"

namespace xt::dropouts
{
    torch::Tensor auto_dropout(torch::Tensor x);

    struct AutoDropout : xt::Module
    {
    public:
        explicit AutoDropout(c10::IntArrayRef probability_shape = {}, double initial_dropout_rate = 0.05);

    private:
        torch::Tensor log_alpha_;
    };
}
