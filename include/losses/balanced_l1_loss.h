#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor balanced_l1_loss(torch::Tensor x);
    class BalancedL1Loss : xt::Module
    {
    public:
    private:
    };
}
