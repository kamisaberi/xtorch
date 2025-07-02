#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor mish(torch::Tensor x);

    struct Mish : xt::Module {
    public:
        Mish() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}



