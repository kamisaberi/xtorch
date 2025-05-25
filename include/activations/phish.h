#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor phish(torch::Tensor x);

    struct Phish : xt::Module {
    public:
        Phish() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



