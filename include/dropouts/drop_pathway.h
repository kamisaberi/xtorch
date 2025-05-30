#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor drop_pathway(torch::Tensor x);

    struct DropPathway : xt::Module {
    public:
        DropPathway() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



