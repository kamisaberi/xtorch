#pragma once

#include "include/transforms/common.h"

namespace xt::transforms::image
{
    class RandomChoice final : public xt::Module
    {
    public:
        RandomChoice();
        explicit RandomChoice(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
