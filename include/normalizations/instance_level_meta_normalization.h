#pragma once

#include "common.h"

namespace xt::norm
{
    struct InstanceLevelMetaNorm : xt::Module
    {
    public:
        InstanceLevelMetaNorm() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
