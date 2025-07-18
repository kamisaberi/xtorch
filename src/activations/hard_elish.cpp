#include <activations/hard_elish.h>

namespace xt::activations
{
    torch::Tensor hard_elish(const torch::Tensor& x)
    {
        torch::Tensor positive_part = x * torch::clamp(x + 1.0, 0.0, 1.0);
        torch::Tensor negative_part = (torch::exp(x) - 1.0) * torch::clamp(x + 1.0, 0.0, 1.0);

        torch::Tensor result = torch::where(x >= 0, positive_part, negative_part);
        return result;
    }

    auto HardELiSH::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::hard_elish(torch::zeros(10));
    }
}
