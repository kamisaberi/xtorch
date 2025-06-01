#include "include/transforms/signal/inverse_mel_scale.h"

namespace xt::transforms::signal
{
    InverseMelScale::InverseMelScale() = default;

    InverseMelScale::InverseMelScale(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto InverseMelScale::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
