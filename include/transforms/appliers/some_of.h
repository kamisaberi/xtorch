#pragma once

#include "transforms/common.h"

namespace xt::transforms
{
    class SomeOf : xt::Module
    {
    public:
        using TransformFunc = std::function<torch::Tensor(torch::Tensor)>;
        SomeOf();
        explicit SomeOf(std::vector<TransformFunc> transforms);
        torch::Tensor forward(torch::Tensor input) const override;

    private:
        std::vector<TransformFunc> transforms;
    };
}
