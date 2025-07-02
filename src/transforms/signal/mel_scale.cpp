#include "include/transforms/signal/mel_scale.h"

namespace xt::transforms::signal
{
    MelScale::MelScale() = default;

    MelScale::MelScale(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto MelScale::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
