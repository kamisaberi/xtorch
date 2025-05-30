#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor am_lines(torch::Tensor x);

    struct AMLines : xt::Module {
    public:
        AMLines() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



