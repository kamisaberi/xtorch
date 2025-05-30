#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor dra(torch::Tensor x);

    struct DRA : xt::Module {
    public:
        DRA() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



