#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class Brightness final : public xt::Module
    {
    public:
        Brightness();
        explicit Brightness(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
