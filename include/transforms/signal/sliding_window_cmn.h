#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class SlidingWindowCMN final : public xt::Module
    {
    public:
        SlidingWindowCMN();
        explicit SlidingWindowCMN(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
