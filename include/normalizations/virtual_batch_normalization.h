#pragma once

#include "common.h"

namespace xt::norm
{
    struct VirtualBatchNorm : xt::Module
    {
    public:
        VirtualBatchNorm() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
