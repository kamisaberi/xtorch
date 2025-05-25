#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor splash(torch::Tensor x);

    struct SPLASH : xt::Module {
    public:
        SPLASH() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



