#pragma once

#include "common.h"

namespace xt::norm
{
    struct FilterResponseNorm : xt::Module
    {
    public:
        FilterResponseNorm() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
