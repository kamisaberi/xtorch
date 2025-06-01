#include "include/transforms/signal/background_noise_addition.h"

namespace xt::transforms::signal
{
    BackgroundNoiseAddition::BackgroundNoiseAddition() = default;

    BackgroundNoiseAddition::BackgroundNoiseAddition(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto BackgroundNoiseAddition::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
