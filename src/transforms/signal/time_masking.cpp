#include "include/transforms/signal/time_masking.h"

namespace xt::transforms::signal
{
    TimeMasking::TimeMasking() = default;

    TimeMasking::TimeMasking(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto TimeMasking::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
