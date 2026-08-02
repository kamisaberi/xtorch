#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor triplet_entropy_loss(torch::Tensor x);
    class TripletEntropyLoss : xt::Module
    {
    public:


    private:
    };
}
