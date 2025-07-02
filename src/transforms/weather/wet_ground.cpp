#include "include/transforms/weather/wet_ground.h"

namespace xt::transforms::weather
{
    WetGround::WetGround() = default;

    WetGround::WetGround(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto WetGround::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
