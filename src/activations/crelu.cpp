#include "include/activations/crelu.h"

namespace xt::activations
{
    torch::Tensor crelu(torch::Tensor x)
    {
    }

    torch::Tensor CReLU::forward(torch::Tensor x) const
    {
        return xt::activations::crelu(x);
    }
}
