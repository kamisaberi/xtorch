#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor sensor_dropout(torch::Tensor x);

    struct SensorDropout : xt::Module {
    public:
        SensorDropout() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



