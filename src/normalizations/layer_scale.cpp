#include "include/normalizations/layer_scale.h"




#include <torch/torch.h>
#include <iostream>
#include <vector>

// Forward declaration for the Impl struct
struct LayerScaleImpl;

// The main module struct that users will interact with.
struct LayerScale : torch::nn::ModuleHolder<LayerScaleImpl> {
    using torch::nn::ModuleHolder<LayerScaleImpl>::ModuleHolder;

    torch::Tensor forward(torch::Tensor x) {
        return impl_->forward(x);
    }
};

// The implementation struct for LayerScale
struct LayerScaleImpl : torch::nn::Module {
    int64_t dim_;           // The feature dimension (number of channels)
    double initial_value_;  // Initial value for the learnable scaling factors

    // Learnable scaling parameter (lambda)
    // It's a vector of size 'dim_', one scale factor per channel.
    torch::Tensor lambda_;

    LayerScaleImpl(int64_t dim, double initial_value = 1e-4)
        : dim_(dim),
          initial_value_(initial_value) {

        TORCH_CHECK(dim > 0, "Dimension 'dim' must be positive.");

        // Initialize lambda_ as a learnable parameter vector of size 'dim_'.
        // All elements are initialized to initial_value_.
        lambda_ = register_parameter("lambda", torch::full({dim_}, initial_value_));
    }

    torch::Tensor forward(torch::Tensor x) {
        // Input x can have various shapes, but its channel/feature dimension
        // must match this->dim_.
        // Common shapes:
        // - (Batch, NumTokens, Channels) for Transformers (dim_ = Channels, lambda applied to last dim)
        // - (Batch, Channels, Height, Width) for CNNs (dim_ = Channels, lambda applied to dim 1)

        // We need to ensure lambda_ broadcasts correctly with x.
        // lambda_ is 1D: (dim_).
        // If x is (N, L, C) and dim_ == C, lambda should be (1, 1, C) for broadcasting.
        // If x is (N, C, H, W) and dim_ == C, lambda should be (1, C, 1, 1) for broadcasting.

        TORCH_CHECK(x.size(-1) == dim_ || (x.dim() > 1 && x.size(1) == dim_),
            "The size of the feature dimension of input x does not match LayerScale dim. "
            "Input feature dim size: ", (x.dim() > 1 && x.size(1) == dim_ ? x.size(1) : x.size(-1)),
            ", LayerScale dim: ", dim_, ", Input shape: ", x.sizes());


        std::vector<int64_t> lambda_shape_for_broadcast(x.dim(), 1);

        if (x.size(-1) == dim_) {
            // Assume feature dimension is the last one, e.g., (N, L, C)
            lambda_shape_for_broadcast[x.dim() - 1] = dim_;
        } else if (x.dim() > 1 && x.size(1) == dim_) {
            // Assume feature dimension is the second one (channel dim for NCHW), e.g., (N, C, H, W)
            lambda_shape_for_broadcast[1] = dim_;
        } else {
            // This case should be caught by the TORCH_CHECK above, but as a fallback:
            TORCH_CHECK(false, "Could not determine broadcasting shape for lambda. Input shape: ", x.sizes(), " LayerScale dim: ", dim_);
        }

        return x * lambda_.view(lambda_shape_for_broadcast);
    }

    void pretty_print(std::ostream& stream) const override {
        stream << "LayerScale(dim=" << dim_
               << ", initial_value=" << initial_value_ << ")";
    }
};
TORCH_MODULE(LayerScale);


// --- Example Usage ---
int main() {
    torch::manual_seed(0);

    // --- Test Case 1: Transformer-like input (Batch, NumTokens, Channels) ---
    std::cout << "--- Test Case 1: Transformer-like input (N, L, C) ---" << std::endl;
    int64_t N1 = 4, L1 = 10, C1 = 32; // Channels = dim for LayerScale
    double init_val1 = 1e-5;
    LayerScale ls_module1(C1, init_val1);
    // std::cout << ls_module1 << std::endl;

    torch::Tensor x1 = torch::randn({N1, L1, C1});
    std::cout << "Input x1 shape: " << x1.sizes() << std::endl;
    std::cout << "Initial lambda values (first 5): " << ls_module1->lambda_.slice(0, 0, 5) << std::endl;

    torch::Tensor y1 = ls_module1->forward(x1);
    std::cout << "Output y1 shape: " << y1.sizes() << std::endl;

    // Check if output is scaled input
    torch::Tensor expected_y1_val = x1.select(0,0).select(0,0).select(0,0) * init_val1; // x[0,0,0] * lambda[0]
    std::cout << "y1[0,0,0] (actual): " << y1[0][0][0].item<double>()
              << ", (expected): " << expected_y1_val.item<double>() << std::endl;
    TORCH_CHECK(torch::allclose(y1[0][0][0], expected_y1_val), "Output value mismatch for Test Case 1.");
    TORCH_CHECK(y1.sizes() == x1.sizes(), "Output y1 shape mismatch!");


    // --- Test Case 2: CNN-like input (Batch, Channels, Height, Width) ---
    std::cout << "\n--- Test Case 2: CNN-like input (N, C, H, W) ---" << std::endl;
    int64_t N2 = 2, C2 = 3, H2 = 8, W2 = 8; // Channels = dim for LayerScale
    double init_val2 = 0.1; // Using a larger init_val for easier visual check
    LayerScale ls_module2(C2, init_val2);
    // std::cout << ls_module2 << std::endl;

    torch::Tensor x2 = torch::ones({N2, C2, H2, W2}); // Input of ones
    std::cout << "Input x2 shape: " << x2.sizes() << std::endl;
    std::cout << "Input x2 is all ones." << std::endl;
    std::cout << "Initial lambda values: " << ls_module2->lambda_ << std::endl;


    torch::Tensor y2 = ls_module2->forward(x2);
    std::cout << "Output y2 shape: " << y2.sizes() << std::endl;
    // Since input is ones, output should be init_val2 everywhere for each corresponding channel
    std::cout << "Output y2[0,0,0,0] (should be " << init_val2 << "): " << y2[0][0][0][0].item<double>() << std::endl;
    std::cout << "Output y2[0,1,0,0] (should be " << init_val2 << "): " << y2[0][1][0][0].item<double>() << std::endl;
    TORCH_CHECK(torch::allclose(y2, torch::full_like(x2, init_val2)), "Output y2 is not init_val2 everywhere.");
    TORCH_CHECK(y2.sizes() == x2.sizes(), "Output y2 shape mismatch!");


    // --- Test Case 3: Check backward pass ---
    std::cout << "\n--- Test Case 3: Backward pass check ---" << std::endl;
    LayerScale ls_module3(C1, 1e-6);
    ls_module3->train(); // Ensure lambda has requires_grad=true

    torch::Tensor x3 = torch::randn({N1, L1, C1}, torch::requires_grad());
    torch::Tensor y3 = ls_module3->forward(x3);
    torch::Tensor loss = y3.mean(); // Simple loss
    loss.backward();

    bool grad_exists_x3 = x3.grad().defined() && x3.grad().abs().sum().item<double>() > 0;
    bool grad_exists_lambda = ls_module3->lambda_.grad().defined() &&
                              ls_module3->lambda_.grad().abs().sum().item<double>() > 0;

    std::cout << "Gradient exists for x3: " << (grad_exists_x3 ? "true" : "false") << std::endl;
    std::cout << "Gradient exists for lambda: " << (grad_exists_lambda ? "true" : "false") << std::endl;
    if (grad_exists_lambda) {
        std::cout << "Lambda gradients (first 5): " << ls_module3->lambda_.grad().slice(0, 0, 5) << std::endl;
    }

    TORCH_CHECK(grad_exists_x3, "No gradient for x3!");
    TORCH_CHECK(grad_exists_lambda, "No gradient for lambda!");

    std::cout << "\nLayerScale tests finished." << std::endl;
    return 0;
}



namespace xt::norm
{
    auto LayerScale::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
