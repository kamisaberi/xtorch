#pragma once

#include "common.h"

namespace xt::norm
{
    struct DecorrelatedBatchNorm : xt::Module
    {
    public:
        DecorrelatedBatchNorm() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
