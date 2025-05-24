#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class SomeOf : xt::Module
    {
    public:
        SomeOf();
        explicit SomeOf(std::vector<xt::Module> transforms);
        torch::Tensor forward(torch::Tensor input) const override;

    private:
        std::vector<xt::Module> transforms;
    };
}
