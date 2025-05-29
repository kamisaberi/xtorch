#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class HomogeneousFog final : public xt::Module
    {
    public:
        HomogeneousFog();
        explicit HomogeneousFog(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
