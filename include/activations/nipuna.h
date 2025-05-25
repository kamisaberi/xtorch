#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor nipuna(torch::Tensor x);

    struct Nipuna : xt::Module {
    public:
        Nipuna() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



