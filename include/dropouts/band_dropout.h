#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor band_dropout(torch::Tensor x);

    struct BandDropout : xt::Module {
    public:
        BandDropout() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



