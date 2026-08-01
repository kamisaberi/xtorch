#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor early_exiting_loss(torch::Tensor x);
    class EarlyExitingLoss : xt::Module
    {
    public:
        EarlyExitingLoss() = default;


    private:
    };
}
