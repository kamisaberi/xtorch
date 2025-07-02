#include "include/transforms/target/quantile_transformer.h"

namespace xt::transforms::target
{
    QuantileTransformer::QuantileTransformer() = default;

    QuantileTransformer::QuantileTransformer(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto QuantileTransformer::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
