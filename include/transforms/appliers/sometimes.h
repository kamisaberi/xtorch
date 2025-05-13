#pragma once

#include "../../headers/transforms.h"

namespace xt::transforms
{
    class Sometimes : xt::Module
    {
    public:
        using TransformFunc = std::function<torch::Tensor(torch::Tensor)>;
        Sometimes();
        explicit Sometimes(std::vector<TransformFunc> transforms);
        torch::Tensor forward(torch::Tensor input) const override;

    private:
        std::vector<TransformFunc> transforms;
    };
}
