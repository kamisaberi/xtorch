#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor pmish(torch::Tensor x);

    struct PMish : xt::Module {
    public:
        PMish() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



