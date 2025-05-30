#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor concrete_dropout(torch::Tensor x);

    struct ConcreteDropout : xt::Module {
    public:
        ConcreteDropout() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



