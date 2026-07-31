#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor arcface_loss(torch::Tensor x);

}
