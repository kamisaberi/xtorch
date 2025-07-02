#pragma once
#include "common.h"


namespace xt::losses
{
    torch::Tensor elastic_face(torch::Tensor x);
    class ElasticFace : xt::Module
    {
    public:
        ElasticFace() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
