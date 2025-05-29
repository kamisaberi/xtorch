#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class AccumulatedSnow final : public xt::Module
    {
    public:
        AccumulatedSnow();
        explicit AccumulatedSnow(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
