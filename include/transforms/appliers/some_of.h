#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class SomeOf : xt::Module
    {
    public:
        SomeOf();
        explicit SomeOf(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<torch::Tensor> tensors) -> std::any  override;

    private:
        std::vector<xt::Module> transforms;
    };
}
