#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor arcface_loss(torch::Tensor x);

    class ArcFaceLoss : xt::Module
    {
    public:
        ArcFaceLoss() = default;

    private:
    };
}
