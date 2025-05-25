#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor siren(torch::Tensor x);

    struct Siren : xt::Module {
    public:
        Siren() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



