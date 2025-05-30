#include "include/dropouts/drop_pathway.h"

namespace xt::dropouts
{
    torch::Tensor drop_pathway(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto DropPathway::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::dropouts::drop_pathway(torch::zeros(10));
    }
}
