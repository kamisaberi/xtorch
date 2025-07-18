#include <activations/reglu.h>

namespace xt::activations
{
    torch::Tensor reglu(const torch::Tensor& x, int64_t dim)
    {
        TORCH_CHECK(x.dim() > 0, "Input tensor must have at least one dimension.");
        TORCH_CHECK(dim < x.dim() && dim >= -x.dim(), "Dimension out of range.");
        if (dim < 0)
        {
            dim += x.dim();
        }
        TORCH_CHECK(x.size(dim) % 2 == 0, "Dimension ", dim, " size (", x.size(dim), ") must be even for ReGLU split.");

        auto chunks = torch::chunk(x, 2, dim);
        torch::Tensor x_a = chunks[0];
        torch::Tensor x_b = chunks[1];

        return x_a * torch::relu(x_b);
    }

    auto ReGLU::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::reglu(torch::zeros(10));
    }
}
