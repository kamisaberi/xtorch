#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor nipuna(torch::Tensor x);

    struct Nipuna : xt::Module {
    public:
        Nipuna() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}



