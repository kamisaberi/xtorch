#pragma once

#include "common.h"

namespace xt::dropouts {

    struct EmbeddingDropout : xt::Module {
    public:
        EmbeddingDropout(double p_drop_entire_embedding = 0.1);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
        double p_drop_entire_embedding_; // Probability of dropping an entire embedding vector for a token
        double epsilon_ = 1e-7;           // For numerical stability

    };
}



