#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor dsam_loss(torch::Tensor x);
    class DSAMLoss : xt::Module
    {
    public:

        DSAMLoss() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
