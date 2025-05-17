#pragma once

#include "transforms/common.h"

namespace xt::transforms
{
    class Repeat : public xt::Module
    {
    public:
        Repeat();
        explicit Repeat(xt::Module transform, int n_times);
        torch::Tensor forward(torch::Tensor input) const override;
    private:
        xt::Module transform;
        int n_times_;
    };
}
