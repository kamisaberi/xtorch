#include "include/activations/geglu.h"

namespace xt::activations
{
    torch::Tensor geglu(const torch::Tensor x, int64_t dim = 1) {
        TORCH_CHECK(x.dim() > 0, "Input tensor must have at least one dimension.");
        TORCH_CHECK(dim < x.dim() && dim >= -x.dim(), "Dimension out of range.");
        if (dim < 0) {
            dim += x.dim();
        }
        TORCH_CHECK(x.size(dim) % 2 == 0, "Dimension ", dim, " size (", x.size(dim), ") must be even for GEGLU split.");

        auto chunks = torch::chunk(x, 2, dim);
        torch::Tensor x_a = chunks[0];
        torch::Tensor x_b = chunks[1];

        return x_a * torch::gelu(x_b, "none"); // "none" uses the erf-based GELU
    }

    auto GeGLU::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::geglu(torch::zeros(10));
    }
}
