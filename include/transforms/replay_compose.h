#pragma once

#include "../headers/transforms.h"

namespace xt::transforms
{
    class ReplayCompose : xt::Module
    {
    public:
        using TransformFunc = std::function<torch::Tensor(torch::Tensor)>;
        ReplayCompose();
        explicit ReplayCompose(std::vector<TransformFunc> transforms);
        torch::Tensor operator()(torch::Tensor input) const override;

    private:
        std::vector<TransformFunc> transforms;
    };
}
