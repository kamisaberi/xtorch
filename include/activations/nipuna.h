#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor nipuna(const torch::Tensor& x, double a = 0.25, double b = 0.05);

    struct Nipuna : xt::Module {
    public:
        Nipuna() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}



