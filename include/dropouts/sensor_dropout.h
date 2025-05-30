#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor sensor_dropout(torch::Tensor x);

    struct SensorDropout : xt::Module {
    public:
        SensorDropout() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



