#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor lovasz_softmax(torch::Tensor x);
    class LovaszSoftmax : xt::Module
    {
    public:
        LovaszSoftmax() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}
