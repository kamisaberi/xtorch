#pragma once

#include "../headers/transforms.h"

namespace xt::transforms
{
    class Repeat : xt::Module
    {
    public:
        using TransformFunc = std::function<torch::Tensor(torch::Tensor)>;
        Repeat();
        explicit Repeat(TransformFunc transform);
        torch::Tensor operator()(torch::Tensor input) const;

    private:
        TransformFunc transform;
    };
}
