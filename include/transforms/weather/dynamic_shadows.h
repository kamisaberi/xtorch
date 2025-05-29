#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class DynamicShadows final : public xt::Module
    {
    public:
        DynamicShadows();
        explicit DynamicShadows(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
