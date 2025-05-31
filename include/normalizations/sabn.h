#pragma once

#include "common.h"

namespace xt::norm
{
    struct SABN : xt::Module
    {
    public:
        SABN() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
