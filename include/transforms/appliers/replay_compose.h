#pragma once

#include "../common.h"

namespace xt::transforms
{
    class ReplayCompose : xt::Module
    {
    public:
        ReplayCompose();
        explicit ReplayCompose(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
        std::vector<xt::Module> transforms;
    };
}
