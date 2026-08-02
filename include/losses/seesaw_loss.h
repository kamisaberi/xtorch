#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor seesaw_loss(torch::Tensor x);
    class SeesawLoss : xt::Module
    {
    public:


    private:
    };
}
