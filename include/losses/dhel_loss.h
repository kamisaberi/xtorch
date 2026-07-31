#pragma once
#include "common.h"


namespace xt::losses
{

    torch::Tensor dhel_loss(torch::Tensor x);
    class DHELLoss : xt::Module
    {
    public:

    private:
    };
}
