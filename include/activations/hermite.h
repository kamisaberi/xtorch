#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor hermite(torch::Tensor x);

    struct Hermite : xt::Module {
    public:
        Hermite() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



