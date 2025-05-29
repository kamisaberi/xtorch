#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class PowerTransformer final : public xt::Module
    {
    public:
        PowerTransformer();
        explicit PowerTransformer(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
