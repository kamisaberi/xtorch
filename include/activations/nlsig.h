#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor nlsig(torch::Tensor x);

    struct NLSIG : xt::Module {
    public:
        NLSIG() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



