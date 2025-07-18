#include <activations/nfn.h>

namespace xt::activations
{
    torch::Tensor nfn(
        const torch::Tensor& x,
        const torch::Tensor& alpha, // Shape (num_filters_out, num_filters_in, filter_size)
        const torch::Tensor& beta // Shape (num_filters_out, num_filters_in, filter_size)
    )
    {
        TORCH_CHECK(x.dim() >= 2,
                    "Input x must be at least 2D (e.g., Batch x Features or Batch x Channels x Spatial).");
        TORCH_CHECK(alpha.dim() == 3, "alpha must be 3D (num_filters_out, num_filters_in, filter_size).");
        TORCH_CHECK(beta.dim() == 3, "beta must be 3D (num_filters_out, num_filters_in, filter_size).");
        TORCH_CHECK(alpha.sizes() == beta.sizes(), "alpha and beta must have the same shape.");

        int64_t num_filters_out = alpha.size(0);
        int64_t num_filters_in = alpha.size(1);
        int64_t filter_size = alpha.size(2); // This is 'k' in the paper's notation for filter size

        // The NFN paper applies this in the context of convolutions.
        // For a simple function, we need to define how 'x' relates to 'num_filters_in'.
        // Let's assume 'x' is (Batch, num_filters_in) for a fully connected style application,
        // or (Batch, num_filters_in, Spatial...) for convolutional.
        // The "filter_size" here refers to the width of the frequency filter.
        // The core NFN operation is typically applied after a linear layer (Wx),
        // and then the frequency filtering happens on the *weights* W, not directly on x.
        //
        // The request is for an "activation function." NFNs are more like a layer type.
        // If we interpret this as applying the NFN *non-linearity* component to an input x,
        // assuming x has already been "filtered" or prepared, we can simplify.
        //
        // Let's assume 'x' is the input *to which the NFN-style non-linearity is applied*.
        // The alpha and beta parameters are per-output feature (num_filters_out).
        // The "num_filters_in" and "filter_size" in alpha/beta would then apply if
        // this non-linearity itself was "filter-like" or had internal structure.
        //
        // Given the paper's formula (1): y = alpha * x + beta * sin(alpha * x + beta)
        // where alpha and beta are learnable.
        // For "Network Function Network", these alpha and beta are *functions* of weights of a prior layer.
        //
        // If we are to create a simple *activation function* inspired by NFN's non-linearity,
        // we would take alpha and beta as per-element or per-channel parameters for 'x'.

        // Simpler interpretation: alpha and beta are parameters for each element of x,
        // or broadcastable to x. Let's assume alpha and beta are of the same shape as x, or broadcastable.
        // This deviates from the NFN paper's layer structure but captures the non-linear form.

        // Let's re-interpret based on a simple element-wise activation form:
        // f(x; alpha, beta) = alpha * x + beta * sin(alpha * x + beta)
        // where alpha and beta are learnable parameters, possibly per-channel.
        // For this functional version, alpha and beta will be passed and assumed to be
        // broadcastable to x's shape.

        // For the simple function version, alpha and beta should be broadcastable to x.
        // Let's assume alpha and beta are Tensors with shapes that can broadcast with x.
        TORCH_CHECK(alpha.dim() <= x.dim(),
                    "alpha dimensions should be less than or equal to x dimensions for broadcasting.");
        TORCH_CHECK(beta.dim() <= x.dim(),
                    "beta dimensions should be less than or equal to x dimensions for broadcasting.");


        torch::Tensor term1 = alpha * x;
        torch::Tensor term2_arg = alpha * x + beta; // Inner part of sin
        torch::Tensor term2 = beta * torch::sin(term2_arg);

        return term1 + term2;
    }

    auto NFN::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::activations::nfn(torch::zeros(10), torch::zeros(10), torch::zeros(10));
    }
}
