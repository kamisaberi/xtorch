#pragma once

#include "common.h"

namespace xt::norm
{
    struct LocalContrastNorm : xt::Module
    {
    public:
        LocalContrastNorm() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
