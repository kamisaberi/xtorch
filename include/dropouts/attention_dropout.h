#pragma once

#include "common.h"

namespace xt::dropouts
{
    struct AttentionDropout : xt::Module
    {
    public:

    private:
        double p_; // Probability of an element to be zeroed.
    };
}
