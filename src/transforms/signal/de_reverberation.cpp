#include "include/transforms/signal/de_reverberation.h"

namespace xt::transforms::signal
{
    DeReverberation::DeReverberation() = default;

    DeReverberation::DeReverberation(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto DeReverberation::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
