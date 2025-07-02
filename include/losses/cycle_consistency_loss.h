#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor cycle_consistency_loss(torch::Tensor x);
    class CycleConsistencyLoss : xt::Module
    {
    public:
        CycleConsistencyLoss() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
