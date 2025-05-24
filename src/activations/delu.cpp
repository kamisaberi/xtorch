#include "include/activations/delu.h"

namespace xt::activations
{
    torch::Tensor delu(torch::Tensor x)
    {
    }

    torch::Tensor DELU::forward(torch::Tensor x) const
    {
        return xt::activations::delu(x);
    }
}
