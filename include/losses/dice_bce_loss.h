#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor dice_bce_loss(torch::Tensor x);
}
