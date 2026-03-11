#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor helu(torch::Tensor x, double alpha = 1.0, double lambda_param = 1.0);

    struct HeLU : xt::Module {
    public:


    private:
    };
}



