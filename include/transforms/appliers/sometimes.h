#pragma once

#include "transforms/common.h"

namespace xt::transforms
{
    class Sometimes : xt::Module
    {
    public:
        Sometimes();
        explicit Sometimes(std::vector<xt::Module> transforms);
        torch::Tensor forward(torch::Tensor input) const override;

    private:
        std::vector<xt::Module> transforms;
    };
}
