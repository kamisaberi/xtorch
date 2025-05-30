#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor spectral_dropout(torch::Tensor x);

    struct SpectralDropout : xt::Module {
    public:
        SpectralDropout() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



