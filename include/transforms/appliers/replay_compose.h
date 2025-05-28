#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class ReplayCompose : xt::Module
    {
    public:
        ReplayCompose();
        explicit ReplayCompose(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<torch::Tensor> tensors) -> std::any  override;

    private:
        std::vector<xt::Module> transforms;
    };
}
