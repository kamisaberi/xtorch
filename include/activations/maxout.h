#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor maxout(const torch::Tensor& x, int64_t num_pieces, int64_t dim = 1) ;

    struct Maxout : xt::Module {
    public:
        Maxout() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



