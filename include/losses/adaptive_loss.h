#pragma once

#include "common.h"


namespace xt::losses {
    torch::Tensor adaptive_loss(torch::Tensor x);


    class AdaptiveLoss : xt::Module {
    public:
        AdaptiveLoss() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };


}
