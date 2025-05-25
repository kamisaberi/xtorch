#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor concrete_dropout(torch::Tensor x);

    struct ConcreteDropout : xt::Module {
    public:
        ConcreteDropout() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



