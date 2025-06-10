#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor am_lines(
        const torch::Tensor& x,
        double negative_slope = 0.01,
        double threshold = 1.0,
        double high_positive_slope = 0.5);

    struct AMLines : xt::Module {
    public:
        AMLines() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



