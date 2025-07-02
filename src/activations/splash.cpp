#include "include/activations/splash.h"

namespace xt::activations
{
    torch::Tensor splash(const torch::Tensor& x, double S , double R , double B )
    {
        torch::Tensor log_term_arg = B * torch::pow(torch::abs(S * x), R) + 1.0;
        torch::Tensor result = x * (torch::log(log_term_arg) / torch::log(torch::tensor(2.0, x.options())));
        // log base 2
        return result;
    }

    auto SPLASH::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::splash(torch::zeros(10));
    }
}
