#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor aglu(torch::Tensor x, double s = 1.0);

    struct AGLU : xt::Module {
    public:
        AGLU() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



