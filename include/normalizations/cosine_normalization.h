#pragma once

#include "common.h"

namespace xt::norm
{
    struct CosineNorm : xt::Module
    {
    public:
        CosineNorm() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
