#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor relun(torch::Tensor x);

    struct ReLUN : xt::Module {
    public:
        ReLUN() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}



