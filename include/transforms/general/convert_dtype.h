#pragma once

#include "include/transforms/common.h"

namespace xt::transforms::general {
    class ConvertDType final : public xt::Module {
    public:
        ConvertDType();

        explicit ConvertDType(torch::ScalarType target_dtype);

        torch::Tensor forward(torch::Tensor input) const override;

    private:
        torch::ScalarType target_dtype;
    };
}
