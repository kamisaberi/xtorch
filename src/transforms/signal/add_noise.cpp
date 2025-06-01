#include "include/transforms/signal/add_noise.h"

namespace xt::transforms::signal
{
    AddNoise::AddNoise() = default;

    AddNoise::AddNoise(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto AddNoise::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
