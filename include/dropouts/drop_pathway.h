#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor drop_pathway(torch::Tensor x);

    struct DropPathway : xt::Module {
    public:
        DropPathway() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



