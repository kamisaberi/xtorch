#pragma once
#include "common.h"


namespace xt::losses
{

    torch::Tensor dhel(torch::Tensor x);
    class DHEL : xt::Module
    {
    public:
        DHEL() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
