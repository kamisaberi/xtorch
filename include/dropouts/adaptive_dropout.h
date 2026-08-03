#pragma once

#include "common.h"
#include <torch/torch.h>
#include <vector>
#include <cmath>     // For std::log
#include <ostream>   // For std::ostream


namespace xt::dropouts
{
    struct AdaptiveDropout : xt::Module
    {
    public:
    private:
        torch::Tensor log_alpha_;
    };
}
