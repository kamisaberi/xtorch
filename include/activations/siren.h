#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor siren(torch::Tensor x);

    struct Siren : xt::Module {
    public:
        Siren() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



