#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor flip_loss(torch::Tensor x);
    class FLIPLoss : xt::Module
    {
    public:

        FLIPLoss() = default;

    private:
    };
}
