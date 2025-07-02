#include "include/transforms/image/random_gamma.h"

namespace xt::transforms::image
{
    RandomGamma::RandomGamma() = default;

    RandomGamma::RandomGamma(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto RandomGamma::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
