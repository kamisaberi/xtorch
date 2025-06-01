#include "include/transforms/signal/speed_perturbation.h"

namespace xt::transforms::signal
{
    SpeedPerturbation::SpeedPerturbation() = default;

    SpeedPerturbation::SpeedPerturbation(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto SpeedPerturbation::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
