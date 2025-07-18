#include <activations/star_relu.h>

namespace xt::activations
{
    torch::Tensor star_relu(const torch::Tensor& x, double scale, double bias, double relu_slope,
                            double leaky_slope)
    {
        torch::Tensor x_transformed = scale * x + bias;

        torch::Tensor positive_part = relu_slope * x_transformed;
        torch::Tensor negative_part = leaky_slope * x_transformed;

        torch::Tensor result = torch::where(x_transformed >= 0, positive_part, negative_part);
        return result;
    }

    auto StarReLU::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::star_relu(torch::zeros(10));
    }
}
