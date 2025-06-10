#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor m_arcsinh(const torch::Tensor& x, double m = 1.0) ;

    struct MArcsinh : xt::Module {
    public:
        MArcsinh() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



