#include "include/activations/elish.h"

namespace xt::activations
{
    torch::Tensor elish(torch::Tensor x)
    {
    }

    torch::Tensor ELiSH::forward(torch::Tensor x) const
    {
        return xt::activations::elish(x);
    }
}
