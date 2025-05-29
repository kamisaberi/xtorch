#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class WetGround final : public xt::Module
    {
    public:
        WetGround();
        explicit WetGround(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
