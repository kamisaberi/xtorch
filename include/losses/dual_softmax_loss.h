#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor dual_softmax_loss(torch::Tensor x);
    class DualSoftmaxLoss : xt::Module
    {
    public:


    private:
    };
}
