#include "include/transforms/signal/sliding_window_cmn.h"

namespace xt::transforms::signal
{
    SlidingWindowCMN::SlidingWindowCMN() = default;

    SlidingWindowCMN::SlidingWindowCMN(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto SlidingWindowCMN::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
