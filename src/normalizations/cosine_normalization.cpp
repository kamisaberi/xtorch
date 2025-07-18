#include <normalizations/cosine_normalization.h>


// #include <torch/torch.h>
// #include <iostream>
// #include <vector>
//
// // Forward declaration for the Impl struct
// struct CosineNormalizationImpl;
//
// // The main module struct that users will interact with.
// struct CosineNormalization : torch::nn::ModuleHolder<CosineNormalizationImpl> {
//     using torch::nn::ModuleHolder<CosineNormalizationImpl>::ModuleHolder;
//
//     torch::Tensor forward(torch::Tensor x) {
//         return impl_->forward(x);
//     }
// };
//
// // The implementation struct
// struct CosineNormalizationImpl : torch::nn::Module {
//     int64_t dim_;       // Dimension along which to normalize (typically the feature dimension)
//     double eps_;        // Small epsilon to prevent division by zero
//     bool learnable_tau_; // Whether to include a learnable temperature parameter
//     torch::Tensor tau_;  // Learnable temperature parameter (scalar)
//
//     CosineNormalizationImpl(int64_t dim = 1, double eps = 1e-8, bool learnable_tau = false, double initial_tau = 1.0)
//         : dim_(dim), eps_(eps), learnable_tau_(learnable_tau) {
//
//         if (learnable_tau_) {
//             // Initialize tau as a learnable scalar parameter.
//             // Some papers initialize tau to a specific value, e.g., 1.0 or 20.0
//             tau_ = register_parameter("tau", torch::tensor({initial_tau}));
//         }
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // x: input tensor, e.g., (N, C) or (N, C, H, W)
//         // dim_: The dimension over which to compute the L2 norm and normalize.
//         //       For (N,C) features, dim_ would typically be 1 (the C dimension).
//         //       For (N,C,H,W) and normalizing each spatial feature vector independently,
//         //       dim_ would still usually be 1 (the C dimension).
//
//         TORCH_CHECK(x.dim() > dim_, "Input tensor dimension (", x.dim(), ") must be greater than normalization dimension (", dim_, ").");
//
//         // Calculate L2 norm along the specified dimension
//         // norm_p(p=2, dim=dim_, keepdim=true)
//         torch::Tensor norm = x.norm(2, dim_, /*keepdim=*/true);
//
//         // Normalize x
//         // Add eps to norm to prevent division by zero if norm is very small.
//         torch::Tensor x_normalized = x / (norm + eps_);
//
//         if (learnable_tau_) {
//             // Scale by the learnable temperature tau
//             // tau_ is a scalar, so it will broadcast.
//             return x_normalized * tau_;
//         } else {
//             return x_normalized;
//         }
//     }
//
//     void pretty_print(std::ostream& stream) const override {
//         stream << "CosineNormalization(dim=" << dim_
//                << ", eps=" << eps_
//                << ", learnable_tau=" << (learnable_tau_ ? "true" : "false");
//         if (learnable_tau_ && tau_.defined()) {
//             stream << ", initial_tau_value=" << tau_.item<double>();
//         }
//         stream << ")";
//     }
// };
// TORCH_MODULE(CosineNormalization);
//
//
// // --- Example Usage ---
// int main() {
//     torch::manual_seed(0);
//
//     // --- Test Case 1: 2D input (N, C), normalize along C (dim=1) ---
//     std::cout << "--- Test Case 1: 2D input (N, C), no learnable tau ---" << std::endl;
//     int64_t N1 = 4, C1 = 10;
//     CosineNormalization cos_norm1(/*dim=*/1, /*eps=*/1e-8, /*learnable_tau=*/false);
//     // std::cout << cos_norm1 << std::endl;
//
//     torch::Tensor x1 = torch::randn({N1, C1});
//     std::cout << "Input x1 shape: " << x1.sizes() << std::endl;
//     // std::cout << "Input x1 example row 0 norm: " << x1[0].norm().item<double>() << std::endl;
//
//
//     torch::Tensor y1 = cos_norm1->forward(x1);
//     std::cout << "Output y1 shape: " << y1.sizes() << std::endl;
//
//     // Check L2 norm of each row in y1 (should be close to 1.0)
//     std::cout << "Output y1, L2 norm of row 0 (should be ~1.0): " << y1[0].norm().item<double>() << std::endl;
//     std::cout << "Output y1, L2 norm of row 1 (should be ~1.0): " << y1[1].norm().item<double>() << std::endl;
//     TORCH_CHECK(torch::allclose(y1[0].norm(), torch::tensor(1.0), 1e-5, 1e-7), "Norm of y1[0] is not 1.0");
//
//
//     // --- Test Case 2: 4D input (N, C, H, W), normalize along C (dim=1) ---
//     // This means each (H,W) spatial location will have its C-dimensional feature vector normalized.
//     std::cout << "\n--- Test Case 2: 4D input (N, C, H, W), with learnable tau ---" << std::endl;
//     int64_t N2 = 2, C2 = 3, H2 = 4, W2 = 4;
//     double initial_tau = 5.0;
//     CosineNormalization cos_norm2(/*dim=*/1, /*eps=*/1e-8, /*learnable_tau=*/true, initial_tau);
//     // std::cout << cos_norm2 << std::endl;
//     std::cout << "Initial tau value from module: " << cos_norm2->tau_.item<double>() << std::endl;
//
//
//     torch::Tensor x2 = torch::randn({N2, C2, H2, W2});
//     std::cout << "Input x2 shape: " << x2.sizes() << std::endl;
//
//     torch::Tensor y2 = cos_norm2->forward(x2);
//     std::cout << "Output y2 shape: " << y2.sizes() << std::endl;
//
//     // Check L2 norm of one feature vector at a spatial location (should be ~initial_tau)
//     // y2[batch_idx, :, height_idx, width_idx]
//     auto feature_vector_y2 = y2.select(0,0).slice(1,0,H2).select(2,0).slice(2,0,W2).select(3,0); // y2[0, :, 0, 0]
//     // The line above is a bit complex due to slice, let's simplify selection for a single spatial point's C-dim vector:
//     // y2[batch_idx, ALL_CHANNELS, spatial_idx_1, spatial_idx_2, ...]
//     // For (N,C,H,W) and dim=1 (channel): we normalize vectors x[n,:,h,w]
//     // So, y2[n,:,h,w] should have norm `tau_`
//     auto one_spatial_vector_y2 = y2.select(0,0).select(2,0).select(3,0); // y2[0, :, 0, 0] of shape (C2)
//     std::cout << "Output y2, L2 norm of feature vector at [0,:,0,0] (should be ~initial_tau=" << initial_tau << "): "
//               << one_spatial_vector_y2.norm().item<double>() << std::endl;
//     TORCH_CHECK(torch::allclose(one_spatial_vector_y2.norm(), torch::tensor(initial_tau), 1e-5, 1e-7),
//                 "Norm of spatial vector in y2 is not initial_tau");
//
//
//     // --- Test Case 3: Check backward pass with learnable tau ---
//     std::cout << "\n--- Test Case 3: Backward pass check with learnable tau ---" << std::endl;
//     CosineNormalization cos_norm3(/*dim=*/1, /*eps=*/1e-8, /*learnable_tau=*/true, /*initial_tau=*/10.0);
//     cos_norm3->train(); // Ensure tau has requires_grad=true
//
//     torch::Tensor x3 = torch::randn({N1, C1}, torch::requires_grad());
//     torch::Tensor y3 = cos_norm3->forward(x3);
//     torch::Tensor loss = y3.mean();
//     loss.backward();
//
//     bool grad_exists_x3 = x3.grad().defined() && x3.grad().abs().sum().item<double>() > 0;
//     bool grad_exists_tau = cos_norm3->tau_.grad().defined() && cos_norm3->tau_.grad().abs().sum().item<double>() > 0;
//
//     std::cout << "Gradient exists for x3: " << (grad_exists_x3 ? "true" : "false") << std::endl;
//     std::cout << "Gradient exists for tau: " << (grad_exists_tau ? "true" : "false") << std::endl;
//     if (grad_exists_tau) {
//         std::cout << "tau.grad: " << cos_norm3->tau_.grad().item<double>() << std::endl;
//     }
//
//     TORCH_CHECK(grad_exists_x3, "No gradient for x3!");
//     TORCH_CHECK(grad_exists_tau, "No gradient for tau!");
//
//     std::cout << "\nCosineNormalization tests finished." << std::endl;
//     return 0;
// }


namespace xt::norm
{
    CosineNorm::CosineNorm(int64_t dim, double eps, bool learnable_tau, double initial_tau)
        : dim_(dim), eps_(eps), learnable_tau_(learnable_tau)
    {
        if (learnable_tau_)
        {
            // Initialize tau as a learnable scalar parameter.
            // Some papers initialize tau to a specific value, e.g., 1.0 or 20.0
            tau_ = register_parameter("tau", torch::tensor({initial_tau}));
        }
    }

    auto CosineNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        vector<std::any> tensors_ = tensors;
        auto x = std::any_cast<torch::Tensor>(tensors_[0]);


        // x: input tensor, e.g., (N, C) or (N, C, H, W)
        // dim_: The dimension over which to compute the L2 norm and normalize.
        //       For (N,C) features, dim_ would typically be 1 (the C dimension).
        //       For (N,C,H,W) and normalizing each spatial feature vector independently,
        //       dim_ would still usually be 1 (the C dimension).

        TORCH_CHECK(x.dim() > dim_, "Input tensor dimension (", x.dim(),
                    ") must be greater than normalization dimension (", dim_, ").");

        // Calculate L2 norm along the specified dimension
        // norm_p(p=2, dim=dim_, keepdim=true)
        torch::Tensor norm = x.norm(2, dim_, /*keepdim=*/true);

        // Normalize x
        // Add eps to norm to prevent division by zero if norm is very small.
        torch::Tensor x_normalized = x / (norm + eps_);

        if (learnable_tau_)
        {
            // Scale by the learnable temperature tau
            // tau_ is a scalar, so it will broadcast.
            return x_normalized * tau_;
        }
        else
        {
            return x_normalized;
        }
    }
}
