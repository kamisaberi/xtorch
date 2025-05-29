#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class Truncate final : public xt::Module
    {
    public:
        Truncate();
        explicit Truncate(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
