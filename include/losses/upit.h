#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor upit(torch::Tensor x);
    class UPIT : xt::Module
    {
    public:
        UPIT() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
