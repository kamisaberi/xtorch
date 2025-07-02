//TODO SHOULD IMPLEMENT
#include "include/activations/andhra.h"

namespace xt::activations
{
    torch::Tensor andhra(const torch::Tensor x, double alpha, double beta)
    {
        torch::Tensor alpha_x = alpha * x;
        torch::Tensor sig_alpha_x = torch::sigmoid(alpha_x);
        torch::Tensor exp_neg_alpha_x = torch::exp(-alpha_x);

        torch::Tensor term_in_bracket = sig_alpha_x + exp_neg_alpha_x * (1.0 - sig_alpha_x);
        torch::Tensor result = (1.0 / alpha) * term_in_bracket - beta * exp_neg_alpha_x;

        return result;
    }

    auto ANDHRA::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::andhra(torch::zeros(10));
    }
}
