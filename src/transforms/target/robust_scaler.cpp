#include "include/transforms/target/robust_scaler.h"

namespace xt::transforms::target
{
    RobustScaler::RobustScaler() = default;

    RobustScaler::RobustScaler(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto RobustScaler::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
