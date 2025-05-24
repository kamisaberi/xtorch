#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor ahaf(torch::Tensor x);

    struct AHAF : xt::Module {
    public:
        AHAF() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



