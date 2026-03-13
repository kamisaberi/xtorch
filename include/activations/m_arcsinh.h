#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor m_arcsinh(const torch::Tensor& x, double m = 1.0) ;

    struct MArcsinh : xt::Module {
    public:

    private:
    };
}



