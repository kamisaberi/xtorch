#include "include/transforms/image/random_grayscale.h"

namespace xt::transforms::image
{
    RandomGrayscale::RandomGrayscale() = default;

    RandomGrayscale::RandomGrayscale(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto RandomGrayscale::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
