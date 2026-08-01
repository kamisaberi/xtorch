#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor gan_least_squares_loss(torch::Tensor x);
    class GANLeastSquaresLoss : xt::Module
    {
    public:


    private:
    };
}
