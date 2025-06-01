#include "include/transforms/image/random_choice.h"

namespace xt::transforms::image
{
    RandomChoice::RandomChoice() = default;

    RandomChoice::RandomChoice(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto RandomChoice::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
