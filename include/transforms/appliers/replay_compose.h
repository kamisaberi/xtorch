#pragma once

#include "transforms/common.h"

namespace xt::transforms
{
    class ReplayCompose : xt::Module
    {
    public:
        using TransformFunc = std::function<torch::Tensor(torch::Tensor)>;
        ReplayCompose();
        explicit ReplayCompose(std::vector<TransformFunc> transforms);
        torch::Tensor forward(torch::Tensor input) const override;

    private:
        std::vector<TransformFunc> transforms;
    };
}
