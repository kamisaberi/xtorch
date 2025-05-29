#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class LongestMaxSize final : public xt::Module
    {
    public:
        LongestMaxSize();
        explicit LongestMaxSize(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
