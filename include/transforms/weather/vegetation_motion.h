#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class VegetationMotion final : public xt::Module
    {
    public:
        VegetationMotion();
        explicit VegetationMotion(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
