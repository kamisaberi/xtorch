#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor multi_loss(torch::Tensor x);
    class MultiLoss : xt::Module
    {
    public:
        MultiLoss() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
