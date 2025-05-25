#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor band_dropout(torch::Tensor x);

    struct BandDropout : xt::Module {
    public:
        BandDropout() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



