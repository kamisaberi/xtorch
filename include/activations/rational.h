#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor rational(torch::Tensor x);

    struct Rational : xt::Module {
    public:
        Rational() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



