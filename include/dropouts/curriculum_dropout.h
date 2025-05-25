#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor curriculum_dropout(torch::Tensor x);

    struct CurriculumDropout : xt::Module {
    public:
        CurriculumDropout() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



