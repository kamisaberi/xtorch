#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor ahaf(torch::Tensor x);

    struct AHAF : xt::Module {
    public:
        AHAF() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



