#include "include/dropouts/sensor_dropout.h"

namespace xt::dropouts
{
    torch::Tensor sensor_dropout(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto SensorDropout::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::dropouts::sensor_dropout(torch::zeros(10));
    }
}
