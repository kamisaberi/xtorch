#include "include/transforms/image/random_order.h"

namespace xt::transforms::image
{
    RandomOrder::RandomOrder() = default;

    RandomOrder::RandomOrder(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto RandomOrder::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
