#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor dsam_loss(torch::Tensor x);
    class DSAMLoss : xt::Module
    {
    public:

    private:
    };
}
