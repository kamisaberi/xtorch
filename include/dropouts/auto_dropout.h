#pragma once

#include "common.h"

namespace xt::dropouts
{
    torch::Tensor auto_dropout(torch::Tensor x);

    struct AutoDropout : xt::Module
    {
    public:

    private:
        torch::Tensor log_alpha_;
    };
}
