#include "include/dropouts/zone_out.h"

namespace xt::dropouts
{
    torch::Tensor zone_out(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto ZoneOut::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::dropouts::zone_out(torch::zeros(10));
    }
}
