#include "include/transforms/signal/vol.h"

namespace xt::transforms::signal
{
    Vol::Vol() = default;

    Vol::Vol(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto Vol::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
