#include "include/normalizations/gradient_normalization.h"



#include <torch/torch.h>
#include <iostream>
#include <vector>

// Forward declaration for the Impl struct
struct HypotheticalGradientInspiredNormImpl;

// The main module struct that users will interact with.
// WARNING: This is NOT standard "Gradient Normalization".
// Standard Gradient Normalization acts on gradients in the backward pass.
// This is a hypothetical forward-pass layer.
struct HypotheticalGradientInspiredNorm : torch::nn::ModuleHolder<HypotheticalGradientInspiredNormImpl> {
    using torch::nn::ModuleHolder<HypotheticalGradientInspiredNormImpl>::ModuleHolder;

    torch::Tensor forward(torch::Tensor x) {
        return impl_->forward(x);
    }
};

// The implementation struct
struct HypotheticalGradientInspiredNormImpl : torch::nn::Module {
    // This layer will normalize the input tensor by its L2 norm,
    // optionally followed by a learnable scaling factor.
    // This is similar to L2 Normalization or Cosine Normalization.

    int64_t dim_;       // Dimension along which to normalize (or -1 for global norm)
    double eps_;        // Small epsilon to prevent division by zero
    bool learnable_scale_; // Whether to include a learnable scaling factor
    torch::Tensor scale_;  // Learnable scaling factor (scalar or per-feature)

    HypotheticalGradientInspiredNormImpl(int64_t dim = -1, // -1 for global norm, or specify a dim
                                         double eps = 1e-8,
                                         bool learnable_scale = true,
                                         double initial_scale = 1.0)
        : dim_(dim), eps_(eps), learnable_scale_(learnable_scale) {

        if (learnable_scale_) {
            if (dim_ == -1 || dim_ >=0 ) { // scalar scale if global norm or if scale applies to whole norm'd tensor
                scale_ = register_parameter("scale", torch::tensor({initial_scale}));
            } else {
                // This case is more complex if scale is per-feature after dim-wise norm.
                // For simplicity, let's assume scalar scale or per-feature if dim indicates feature dim.
                // If dim_ refers to a feature dimension, scale_ could be {num_features}.
                // For now, only scalar scale is implemented for simplicity when learnable_scale is true.
                // A more complete implementation might have scale_ shape depend on dim_.
                TORCH_WARN("Learnable scale with specific dim != -1 might require per-feature scaling. This impl uses scalar scale.");
                 scale_ = register_parameter("scale", torch::tensor({initial_scale}));
            }
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        // x: input tensor

        torch::Tensor norm;
        if (dim_ == -1) {
            // Global L2 norm: sqrt(sum of squares of all elements)
            norm = x.norm(); // Froebenius norm for matrices, L2 for vectors
        } else {
            // L2 norm along a specific dimension
            TORCH_CHECK(x.dim() > dim_ && dim_ >= 0,
                        "Input tensor dimension (", x.dim(), ") must be greater than normalization dimension (", dim_,
                        ") and dim must be non-negative, or -1 for global norm.");
            norm = x.norm(2, dim_, /*keepdim=*/true);
        }

        // Normalize x
        torch::Tensor x_normalized = x / (norm + eps_);

        if (learnable_scale_) {
            // Apply learnable scale
            // scale_ is scalar and will broadcast.
            return x_normalized * scale_;
        } else {
            return x_normalized;
        }
    }

    void pretty_print(std::ostream& stream) const override {
        stream << "HypotheticalGradientInspiredNorm(dim=" << dim_
               << ", eps=" << eps_
               << ", learnable_scale=" << (learnable_scale_ ? "true" : "false");
        if (learnable_scale_ && scale_.defined()) {
            stream << ", initial_scale_value=" << scale_.item<double>();
        }
        stream << ")";
    }
};
TORCH_MODULE(HypotheticalGradientInspiredNorm);


// --- Example Usage of the Hypothetical Layer ---
int main() {
    torch::manual_seed(0);

    std::cout << "WARNING: This module is a HYPOTHETICAL forward-pass layer named 'HypotheticalGradientInspiredNorm'." << std::endl;
    std::cout << "Standard Gradient Normalization acts on gradients in the BACKWARD pass." << std::endl;

    // --- Test Case 1: Global normalization (dim=-1) ---
    std::cout << "\n--- Test Case 1: Global normalization, learnable scale ---" << std::endl;
    HypotheticalGradientInspiredNorm hgn_module1(/*dim=*/-1, /*eps=*/1e-8, /*learnable_scale=*/true, /*initial_scale=*/5.0);
    // std::cout << hgn_module1 << std::endl;

    torch::Tensor x1 = torch::randn({2, 3, 4}); // A 3D tensor
    std::cout << "Input x1 shape: " << x1.sizes() << std::endl;
    std::cout << "Input x1 global L2 norm: " << x1.norm().item<double>() << std::endl;

    torch::Tensor y1 = hgn_module1->forward(x1);
    std::cout << "Output y1 shape: " << y1.sizes() << std::endl;
    std::cout << "Output y1 global L2 norm (should be ~initial_scale=" << hgn_module1->scale_.item<double>() << "): "
              << y1.norm().item<double>() << std::endl;
    TORCH_CHECK(torch::allclose(y1.norm(), hgn_module1->scale_), "Global norm of y1 is not scale_");


    // --- Test Case 2: Normalization along a specific dimension (dim=1 for (N,C) features) ---
    std::cout << "\n--- Test Case 2: Dim-wise normalization (dim=1 for N,C), no learnable scale ---" << std::endl;
    int64_t N2 = 4, C2 = 10;
    HypotheticalGradientInspiredNorm hgn_module2(/*dim=*/1, /*eps=*/1e-8, /*learnable_scale=*/false);
    // std::cout << hgn_module2 << std::endl;

    torch::Tensor x2 = torch::randn({N2, C2});
    std::cout << "Input x2 shape: " << x2.sizes() << std::endl;

    torch::Tensor y2 = hgn_module2->forward(x2);
    std::cout << "Output y2 shape: " << y2.sizes() << std::endl;
    // Check L2 norm of each row in y2 (should be ~1.0)
    std::cout << "Output y2, L2 norm of row 0 (should be ~1.0): " << y2[0].norm().item<double>() << std::endl;
    TORCH_CHECK(torch::allclose(y2[0].norm(), torch::tensor(1.0), 1e-5, 1e-7), "Norm of y2[0] is not 1.0");


    // --- Test Case 3: Backward pass check with learnable scale ---
    std::cout << "\n--- Test Case 3: Backward pass check (global norm, learnable scale) ---" << std::endl;
    HypotheticalGradientInspiredNorm hgn_module3(/*dim=*/-1, 1e-8, true, 1.0);
    hgn_module3->train();

    torch::Tensor x3 = torch::randn({N2, C2}, torch::requires_grad());
    torch::Tensor y3 = hgn_module3->forward(x3);
    torch::Tensor loss = y3.mean();
    loss.backward();

    bool grad_exists_x3 = x3.grad().defined() && x3.grad().abs().sum().item<double>() > 0;
    bool grad_exists_scale = hgn_module3->scale_.grad().defined() &&
                             hgn_module3->scale_.grad().abs().sum().item<double>() > 0;

    std::cout << "Gradient exists for x3: " << (grad_exists_x3 ? "true" : "false") << std::endl;
    std::cout << "Gradient exists for scale: " << (grad_exists_scale ? "true" : "false") << std::endl;

    TORCH_CHECK(grad_exists_x3, "No gradient for x3!");
    TORCH_CHECK(grad_exists_scale, "No gradient for scale!");

    std::cout << "\nHypotheticalGradientInspiredNorm tests finished." << std::endl;


    std::cout << "\n--- Example of True Gradient Normalization (Clipping) in LibTorch (NOT part of the module) ---" << std::endl;
    // This is how you would typically do gradient clipping (a form of gradient normalization)
    torch::nn::Linear linear_layer(10, 5);
    torch::Tensor input_for_clip = torch::randn({4, 10});
    torch::Tensor target_for_clip = torch::randn({4, 5});
    torch::Tensor output_for_clip = linear_layer->forward(input_for_clip);
    torch::Tensor loss_for_clip = torch::mse_loss(output_for_clip, target_for_clip);

    loss_for_clip.backward(); // Compute gradients

    // Before optimizer step, clip gradients
    double max_norm = 1.0;
    torch::nn::utils::clip_grad_norm_(linear_layer->parameters(), max_norm);
    std::cout << "Gradients clipped by norm " << max_norm << " for linear_layer parameters." << std::endl;
    // Now an optimizer would use these (potentially clipped) gradients
    // torch::optim::SGD optimizer(linear_layer->parameters(), 0.1);
    // optimizer.step();


    return 0;
}



namespace xt::norm
{
    auto GradientNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
