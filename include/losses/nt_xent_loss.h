#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor nt_xent(torch::Tensor x);
    class NTXent : xt::Module
    {
    public:
        NTXent() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
