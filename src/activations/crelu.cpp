#include "include/activations/crelu.h"

namespace xt::activations
{
    torch::Tensor crelu(const torch::Tensor& x, int64_t dim)
    {
        torch::Tensor relu_x = torch::relu(x);
        torch::Tensor relu_neg_x = torch::relu(-x);
        torch::Tensor result = torch::cat({relu_x, relu_neg_x}, dim);
        return result;
    }


    auto CReLU::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::crelu(torch::zeros(10));
    }
}
