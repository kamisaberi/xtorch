#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor zone_out(torch::Tensor x);

    struct ZoneOut : xt::Module {
    public:
        ZoneOut() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



