#include <activations/swigelu.h>

namespace xt::activations
{
    torch::Tensor swiglu(const torch::Tensor& x, int64_t dim, double beta)
    {
        TORCH_CHECK(x.dim() > 0, "Input tensor must have at least one dimension.");
        TORCH_CHECK(dim < x.dim() && dim >= -x.dim(), "Dimension out of range.");
        if (dim < 0)
        {
            dim += x.dim();
        }
        TORCH_CHECK(x.size(dim) % 2 == 0, "Dimension ", dim, " size (", x.size(dim),
                    ") must be even for SwiGLU split.");

        auto chunks = torch::chunk(x, 2, dim);
        torch::Tensor x_a = chunks[0];
        torch::Tensor x_b = chunks[1];

        return x_a * (x_b * torch::sigmoid(beta * x_b)); // x_b * sigmoid(beta * x_b) is Swish(x_b)
    }

    auto SwiGLU::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::swiglu(torch::zeros(10));
    }
}
