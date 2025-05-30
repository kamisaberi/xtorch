#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor splash(torch::Tensor x);

    struct SPLASH : xt::Module {
    public:
        SPLASH() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



