#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor rank_based_loss(torch::Tensor x);
    class RankBasedLoss : xt::Module
    {
    public:
        RankBasedLoss() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
