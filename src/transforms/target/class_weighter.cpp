#include "include/transforms/target/class_weighter.h"

namespace xt::transforms::target
{
    ClassWeighter::ClassWeighter() = default;

    ClassWeighter::ClassWeighter(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto ClassWeighter::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
