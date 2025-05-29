#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class TargetEncoder final : public xt::Module
    {
    public:
        TargetEncoder();
        explicit TargetEncoder(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
