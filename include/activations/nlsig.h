#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor nlsig(torch::Tensor x);

    struct NLSIG : xt::Module {
    public:
        NLSIG() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



