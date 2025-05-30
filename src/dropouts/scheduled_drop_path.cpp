#include "include/dropouts/scheduled_drop_path.h"

namespace xt::dropouts
{
    torch::Tensor scheduled_drop_path(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto ScheduledDropPath::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::dropouts::scheduled_drop_path(torch::zeros(10));
    }
}
