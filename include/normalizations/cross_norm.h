#pragma once

#include "common.h"

namespace xt::norm
{
    struct CrossNorm : xt::Module
    {
    public:
        CrossNorm() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
