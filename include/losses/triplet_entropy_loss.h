#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor triplet_entropy_loss(torch::Tensor x);
    class TripletEntropyLoss : xt::Module
    {
    public:
        TripletEntropyLoss() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}
