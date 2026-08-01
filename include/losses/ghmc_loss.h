#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor ghmc_loss(torch::Tensor x);
    class GHMCLoss : xt::Module
    {
    public:

    private:
    };
}
