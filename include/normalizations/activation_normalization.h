#pragma once

#include "common.h"

namespace xt::norm
{
    struct ActiveNorm : xt::Module
    {
    public:
        ActiveNorm() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
