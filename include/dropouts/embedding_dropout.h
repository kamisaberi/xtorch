#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor embedding_dropout(torch::Tensor x);

    struct EmbeddingDropout : xt::Module {
    public:
        EmbeddingDropout() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



