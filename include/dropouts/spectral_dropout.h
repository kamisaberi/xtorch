#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor spectral_dropout(torch::Tensor x);

    struct SpectralDropout : xt::Module {
    public:
        SpectralDropout() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



