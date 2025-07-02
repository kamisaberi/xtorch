#include "include/transforms/signal/time_stretch.h"

namespace xt::transforms::signal
{
    TimeStretch::TimeStretch() = default;

    TimeStretch::TimeStretch(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto TimeStretch::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
