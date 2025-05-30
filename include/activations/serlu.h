#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor serlu(torch::Tensor x);

    struct SERLU : xt::Module {
    public:
        SERLU() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}



