#pragma once

#include "common.h"

namespace xt::dropouts {
    torch::Tensor embedding_dropout(torch::Tensor x);

    struct EmbeddingDropout : xt::Module {
    public:
        EmbeddingDropout() = default;

        torch::Tensor forward(torch::Tensor x) const override;

    private:
    };
}



