#pragma once

#include "include/transforms/common.h"

namespace xt::transforms::image
{
    class BlackWhite final : public xt::Module
    {
    public:
        BlackWhite();
        explicit BlackWhite(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
