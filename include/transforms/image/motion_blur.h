#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class MotionNlur final : public xt::Module
    {
    public:
        MotionNlur();
        explicit MotionNlur(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
