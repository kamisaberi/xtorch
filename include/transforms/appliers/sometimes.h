#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class Sometimes : xt::Module
    {
    public:
        Sometimes();
        explicit Sometimes(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
        std::vector<xt::Module> transforms;
    };
}
