#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor nfn(torch::Tensor x);

    struct NFN : xt::Module {
    public:
        NFN() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



