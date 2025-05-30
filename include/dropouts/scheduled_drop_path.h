#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor scheduled_drop_path(torch::Tensor x);

    struct ScheduledDropPath : xt::Module {
    public:
        ScheduledDropPath() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



