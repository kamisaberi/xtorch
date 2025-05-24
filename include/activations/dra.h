#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor dra(torch::Tensor x);

    struct DRA : xt::Module {
    public:
        DRA() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



