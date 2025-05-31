#include "include/losses/elastic_face.h"

namespace xt::losses
{
    torch::Tensor elastic_face(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto ElasticFace::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::elastic_face(torch::zeros(10));
    }
}
