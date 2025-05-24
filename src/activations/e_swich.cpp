#include "include/activations/e_swich.h"

namespace xt::activations
{
    torch::Tensor e_swish(torch::Tensor x)
    {
    }

    torch::Tensor ESwish::forward(torch::Tensor x) const
    {
        return xt::activations::e_swish(x);
    }
}
