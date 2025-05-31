#pragma once

#include "common.h"

namespace xt::norm
{
    struct SyncBN : xt::Module
    {
    public:
        SyncBN() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
