#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor aglu(torch::Tensor x);

    struct AGLU : xt::Module {
    public:
        AGLU() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



