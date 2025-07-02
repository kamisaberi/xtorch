#include "include/transforms/target/min_max_scaler.h"

namespace xt::transforms::target
{
    MinMaxScaler::MinMaxScaler() = default;

    MinMaxScaler::MinMaxScaler(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto MinMaxScaler::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
