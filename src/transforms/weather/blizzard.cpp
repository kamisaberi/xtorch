#include "include/transforms/weather/blizzard.h"

namespace xt::transforms::weather
{
    Blizzard::Blizzard() = default;

    Blizzard::Blizzard(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto Blizzard::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
