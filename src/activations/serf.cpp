#include <activations/serf.h>

namespace xt::activations
{
    // const double SQRT_2_PI = std::sqrt(2.0 / M_PI); // For erf derivative-like part

    torch::Tensor serf(const torch::Tensor& x, double k_param, double lambda_param)
    {
        torch::Tensor positive_part = x * torch::erf(lambda_param * torch::log1p(torch::exp(x)));
        // x * erf(lambda * softplus(x))
        torch::Tensor negative_part = x * torch::exp(-k_param * x * x);

        torch::Tensor result = torch::where(x >= 0, positive_part, negative_part);
        return result;
    }

    auto Serf::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::serf(torch::zeros(10));
    }
}
