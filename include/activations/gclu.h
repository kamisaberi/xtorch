#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor gclu(torch::Tensor x, int64_t dim = 1);

    struct GCLU : xt::Module {
    public:
        GCLU() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



