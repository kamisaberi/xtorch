#pragma once

#include "transforms/common.h"

namespace xt::transforms
{
    class ReplayCompose : xt::Module
    {
    public:
        ReplayCompose();
        explicit ReplayCompose(std::vector<xt::Module> transforms);
        torch::Tensor forward(torch::Tensor input) const override;

    private:
        std::vector<xt::Module> transforms;
    };
}
