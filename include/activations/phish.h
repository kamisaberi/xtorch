#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor phish(torch::Tensor x);

    struct Phish : xt::Module {
    public:
        Phish() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



