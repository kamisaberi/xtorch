#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class TimeMasking final : public xt::Module
    {
    public:
        TimeMasking();
        explicit TimeMasking(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
