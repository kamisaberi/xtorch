#include "include/dropouts/embedding_dropout.h"

namespace xt::dropouts
{
    torch::Tensor embedding_dropout(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto EmbeddingDropout::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::dropouts::embedding_dropout(torch::zeros(10));
    }
}
