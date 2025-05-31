#pragma once

#include "common.h"

namespace xt::norm
{
    struct ConditionalBatchNorm : xt::Module
    {
    public:
        ConditionalBatchNorm() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
