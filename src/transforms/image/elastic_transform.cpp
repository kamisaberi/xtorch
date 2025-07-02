#include "include/transforms/image/elastic_transform.h"

namespace xt::transforms::image
{
    ElasticTransform::ElasticTransform() = default;

    ElasticTransform::ElasticTransform(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto ElasticTransform::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
