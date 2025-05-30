#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor m_arcsinh(torch::Tensor x);

    struct MArcsinh : xt::Module {
    public:
        MArcsinh() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



