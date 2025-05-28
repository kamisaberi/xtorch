#pragma once

#include "include/transforms/common.h"

namespace xt::transforms::general {
    class ConvertDType final : public xt::Module {
    public:
        ConvertDType();

        explicit ConvertDType(torch::ScalarType target_dtype);

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
        torch::ScalarType target_dtype;
    };
}
