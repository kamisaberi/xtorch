#pragma once

#include "common.h"

namespace xt::norm
{
    struct SPADE : xt::Module
    {
    public:
        SPADE() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
