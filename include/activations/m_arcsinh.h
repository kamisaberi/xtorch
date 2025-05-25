#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor m_arcsinh(torch::Tensor x);

    struct MArcsinh : xt::Module {
    public:
        MArcsinh() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



