#include "include/activations/smooth_step.h"

namespace xt::activations
{
    torch::Tensor smooth_step(const torch::Tensor& x, double edge0, double edge1)
    {
        TORCH_CHECK(edge0 < edge1, "edge0 must be less than edge1 for smoothstep.");

        torch::Tensor t = torch::clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
        torch::Tensor result = t * t * (3.0 - 2.0 * t);
        return result;
    }

    auto SmoothStep::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::smooth_step(torch::zeros(10));
    }
}
