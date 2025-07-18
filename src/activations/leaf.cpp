#include <activations/leaf.h>

namespace xt::activations
{
    torch::Tensor leaf(const torch::Tensor& x,
                       const torch::Tensor& s_weights, // Shape (L)
                       const torch::Tensor& r_weights, // Shape (L)
                       const torch::Tensor& u_weights // Shape (L)
    )
    {
        TORCH_CHECK(s_weights.dim() == 1, "s_weights must be 1D");
        TORCH_CHECK(r_weights.dim() == 1, "r_weights must be 1D");
        TORCH_CHECK(u_weights.dim() == 1, "u_weights must be 1D");

        int64_t L = s_weights.size(0);
        TORCH_CHECK(L > 0, "Number of terms L must be greater than 0");
        TORCH_CHECK(r_weights.size(0) == L, "r_weights must have L elements");
        TORCH_CHECK(u_weights.size(0) == L, "u_weights must have L elements");

        torch::Tensor x_reshaped = x.unsqueeze(-1); // (N, 1) or (B, ..., 1)

        torch::Tensor s_reshaped = s_weights.view({1, L}); // (1, L)
        torch::Tensor r_reshaped = r_weights.view({1, L}); // (1, L)
        torch::Tensor u_reshaped = u_weights.view({1, L}); // (1, L)

        torch::Tensor terms = s_reshaped * torch::exp(r_reshaped * x_reshaped + u_reshaped);
        torch::Tensor result = torch::sum(terms, -1);

        return result;
    }

    auto LEAF::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::leaf(torch::zeros(10), torch::zeros(10), torch::zeros(10), torch::zeros(10));
    }
}
