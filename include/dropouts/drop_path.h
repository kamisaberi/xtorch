#pragma once

#include "common.h"

namespace xt::dropouts
{

    struct DropPath : xt::Module
    {
    public:
        DropPath(double p_drop = 0.1);

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double p_drop_;
        double epsilon_ = 1e-7; // For numerical stability in division
    };
}
