#include "include/activations/asu.h"

namespace xt::activations
{
    torch::Tensor asu(const torch::Tensor& x, double alpha, double beta, double gamma)
    {
        torch::Tensor beta_x = beta * x;
        torch::Tensor sig_beta_x = torch::sigmoid(beta_x);
        torch::Tensor result = alpha * x * sig_beta_x + gamma;
        return result;
    }

    auto ASU::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::asu(torch::zeros(10));
    }
}
