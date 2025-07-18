#include <activations/gclu.h>

namespace xt::activations
{
    torch::Tensor gclu(const torch::Tensor x, int64_t dim )
    {
        TORCH_CHECK(x.size(dim) % 2 == 0, "Dimension ", dim, " size (", x.size(dim), ") must be even for GCLU split.");

        auto chunks = torch::chunk(x, 2, dim);
        torch::Tensor x_a = chunks[0];
        torch::Tensor x_b = chunks[1];

        return x_a * torch::cos(x_b);
    }

    auto GCLU::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::gclu(torch::zeros(10));
    }
}
