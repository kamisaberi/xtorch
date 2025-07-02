#include "include/activations/rrelu.h"

namespace xt::activations
{
    torch::Tensor rrelu(
        const torch::Tensor& x,
        double lower ,
        double upper ,
        bool training ,
        c10::optional<at::Generator> generator
    )
    {
        if (training)
        {
            at::Generator gen = generator.has_value() ? generator.value() : at::detail::getDefaultCPUGenerator();

            // Generate random numbers with specified dtype (e.g., kFloat64) then cast
            torch::Tensor a_rand = torch::rand(x.sizes(), gen, x.options().dtype(torch::kFloat64));
            a_rand = a_rand * (upper - lower) + lower;
            a_rand = a_rand.to(x.dtype()); // Cast back to original dtype of x

            return torch::where(x >= 0, x, a_rand * x);
        }
        else
        {
            double a_fixed = (lower + upper) / 2.0;
            // Ensure a_fixed is cast to the correct type for multiplication if x is float
            // The static_cast in the original was good.
            // Or create a tensor from a_fixed.
            // Using static_cast is fine for scalar multiplication.
            return torch::where(x >= 0, x, static_cast<float>(a_fixed) * x);
        }
    }

    auto RReLU::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::rrelu(torch::zeros(10));
    }
}
