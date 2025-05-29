#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class Posterize final : public xt::Module
    {
    public:
        Posterize();
        explicit Posterize(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
