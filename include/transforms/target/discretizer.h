#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class Discretizer final : public xt::Module
    {
    public:
        Discretizer();
        explicit Discretizer(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
