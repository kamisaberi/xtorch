#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class Repeat : public xt::Module
    {
    public:
        Repeat();
        explicit Repeat(std::unique_ptr<xt::Module> transform, int n_times);
        torch::Tensor forward(torch::Tensor input) const override;
    private:
        std::unique_ptr<xt::Module> transform;
        int n_times_;
    };
}
