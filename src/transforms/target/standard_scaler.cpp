#include "include/transforms/target/standard_scaler.h"

namespace xt::transforms::target
{
    StandardScaler::StandardScaler() = default;

    StandardScaler::StandardScaler(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto StandardScaler::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
