#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class Blur final : public xt::Module
    {
    public:
        Blur();
        explicit Blur(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
