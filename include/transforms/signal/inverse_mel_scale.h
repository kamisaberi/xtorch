#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class InverseMelScale final : public xt::Module
    {
    public:
        InverseMelScale();
        explicit InverseMelScale(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
