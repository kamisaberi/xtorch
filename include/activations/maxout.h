#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor maxout(torch::Tensor x);

    struct Maxout : xt::Module {
    public:
        Maxout() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



