#pragma once

#include "common.h"

namespace xt::norm
{
    struct InplaceABN : xt::Module
    {
    public:
        InplaceABN() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
