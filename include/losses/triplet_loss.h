#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor triplet_loss(torch::Tensor x);
    class TripletLoss : xt::Module
    {
    public:
        TripletLoss() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}
