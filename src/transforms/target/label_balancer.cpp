#include "include/transforms/target/label_balancer.h"

namespace xt::transforms::target
{
    LabelBalancer::LabelBalancer() = default;

    LabelBalancer::LabelBalancer(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto LabelBalancer::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
