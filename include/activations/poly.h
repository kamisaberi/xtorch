#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor poly(torch::Tensor x);

    struct Poly : xt::Module {
    public:
        Poly() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}



