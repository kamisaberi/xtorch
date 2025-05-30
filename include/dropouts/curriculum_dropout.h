#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor curriculum_dropout(torch::Tensor x);

    struct CurriculumDropout : xt::Module {
    public:
        CurriculumDropout() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



