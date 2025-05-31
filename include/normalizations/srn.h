#pragma once

#include "common.h"

namespace xt::norm
{
    struct SRN : xt::Module
    {
    public:
        SRN() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
