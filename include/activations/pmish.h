#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor pmish(torch::Tensor x);

    struct PMish : xt::Module {
    public:
        PMish() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



