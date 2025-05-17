#pragma once

#include "transforms/common.h"

namespace xt::transforms::general {
    class ConvertDType final : public xt::Module {
    public:
        Compose();

        explicit ConvertDType(torch::ScalarType target_dtype);

        torch::Tensor forward(torch::Tensor input) const override;

    private:
        torch::ScalarType target_dtype;
    };
}
