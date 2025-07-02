#include "include/transforms/signal/griffin_lim.h"

namespace xt::transforms::signal
{
    GriffinLim::GriffinLim() = default;

    GriffinLim::GriffinLim(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto GriffinLim::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
