#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor triplet_loss(torch::Tensor x);
    class TripletLoss : xt::Module
    {
    public:


    private:
    };
}
