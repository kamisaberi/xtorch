#include <activations/hard_swish.h>

namespace xt::activations
{
    torch::Tensor hard_swich(torch::Tensor x)
    {
        torch::Tensor relu6_x_plus_3 = torch::relu6(x + 3.0);
        torch::Tensor result = x * (relu6_x_plus_3 / 6.0);
        return result;
    }

    auto HardSwish::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::hard_swich(torch::zeros(10));
    }
}
