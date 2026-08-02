#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor varifocal_loss(torch::Tensor x);
    class VarifocalLoss : xt::Module
    {
    public:

    private:
    };
}
