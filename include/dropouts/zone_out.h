#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor zone_out(torch::Tensor x);

    struct ZoneOut : xt::Module {
    public:
        ZoneOut() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



