#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class LatenProjection final : public xt::Module
    {
    public:
        LatenProjection();
        explicit LatenProjection(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
