#include "include/transforms/weather/particle_rain.h"

namespace xt::transforms::weather
{
    ParticleRain::ParticleRain() = default;

    ParticleRain::ParticleRain(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto ParticleRain::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
