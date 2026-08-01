#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor dice_loss(torch::Tensor x);
    class DiceLoss : xt::Module
    {
    public:

        DiceLoss() = default;

    private:
    };
}
