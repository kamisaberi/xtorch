#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor golu(torch::Tensor x, double alpha = 1.0, int64_t dim = 1);

    struct GoLU : xt::Module {
    public:
        GoLU() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



