#pragma once

#include "common.h"

namespace xt::activations {
    struct GoLU : xt::Module {
    public:
        GoLU() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



