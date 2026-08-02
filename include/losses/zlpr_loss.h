#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor zlpr_loss(torch::Tensor x);
    class ZLPRLoss : xt::Module
    {
    public:


    private:
    };
}
