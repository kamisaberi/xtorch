#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class TruncationTrick final : public xt::Module
    {
    public:
        TruncationTrick();
        explicit TruncationTrick(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
