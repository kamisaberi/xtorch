#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor scheduled_drop_path(torch::Tensor x);

    struct ScheduledDropPath : xt::Module {
    public:
        ScheduledDropPath() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



