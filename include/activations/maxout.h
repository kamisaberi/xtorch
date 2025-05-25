#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor maxout(torch::Tensor x);

    struct Maxout : xt::Module {
    public:
        Maxout() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



