#include "include/transforms/signal/time_warping.h"

namespace xt::transforms::signal
{
    TimeWarping::TimeWarping() = default;

    TimeWarping::TimeWarping(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto TimeWarping::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
