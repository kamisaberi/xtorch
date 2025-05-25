#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor serlu(torch::Tensor x);

    struct SERLU : xt::Module {
    public:
        SERLU() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



