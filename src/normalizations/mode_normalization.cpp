#include <normalizations/mode_normalization.h>


// #include <torch/torch.h>
// #include <iostream>
// #include <vector>
// #include <map>
//
// // Forward declaration for the Impl struct
// struct ModeNormalizationImpl;
//
// // The main module struct that users will interact with.
// struct ModeNormalization : torch::nn::ModuleHolder<ModeNormalizationImpl> {
//     using torch::nn::ModuleHolder<ModeNormalizationImpl>::ModuleHolder;
//
//     // Forward method takes the input x and a mode_index tensor
//     // mode_index is expected to be a LongTensor of shape (N,) or (1,) if all samples share a mode
//     torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& mode_index) {
//         return impl_->forward(x, mode_index);
//     }
// };
//
// // The implementation struct for ModeNormalization
// struct ModeNormalizationImpl : torch::nn::Module {
//     int64_t num_features_;    // Number of features in input x (channels)
//     int64_t num_modes_;       // Number of different normalization "modes"
//     double eps_in_;           // Epsilon for the base Instance Normalization
//
//     // Learnable affine parameters (gamma and beta) for each mode.
//     // We'll store these as a single tensor and select them, or as separate parameters.
//     // Using a single tensor for embedding-like lookup is common.
//     torch::Tensor gammas_; // Shape (num_modes_, num_features_)
//     torch::Tensor betas_;  // Shape (num_modes_, num_features_)
//
//     // Base normalization (Instance Normalization for this example)
//     // No learnable affine params for the base IN, as they are mode-specific.
//
//     ModeNormalizationImpl(int64_t num_features, int64_t num_modes, double eps_in = 1e-5)
//         : num_features_(num_features),
//           num_modes_(num_modes),
//           eps_in_(eps_in) {
//
//         TORCH_CHECK(num_features > 0, "num_features must be positive.");
//         TORCH_CHECK(num_modes > 0, "num_modes must be positive.");
//
//         // Initialize mode-specific learnable affine parameters
//         // gammas are typically initialized to 1, betas to 0 for each mode.
//         gammas_ = register_parameter("gammas", torch::ones({num_modes_, num_features_}));
//         betas_  = register_parameter("betas",  torch::zeros({num_modes_, num_features_}));
//     }
//
//     torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& mode_index) {
//         // x: input tensor (N, C, D1, D2, ...) where C is num_features_
//         // mode_index: LongTensor of shape (N,) or (N, 1) or scalar if broadcasting.
//         //             Contains integers from 0 to num_modes_ - 1.
//         //             If shape is (N,), each sample in the batch can have a different mode.
//
//         TORCH_CHECK(x.dim() >= 2, "Input tensor x must have at least 2 dimensions (N, C, ...). Got shape ", x.sizes());
//         TORCH_CHECK(x.size(1) == num_features_,
//                     "Input channels mismatch. Expected ", num_features_, ", got ", x.size(1));
//         TORCH_CHECK(mode_index.scalar_type() == torch::kLong, "mode_index must be a LongTensor.");
//         TORCH_CHECK(mode_index.min().item<int64_t>() >= 0 && mode_index.max().item<int64_t>() < num_modes_,
//                     "mode_index values out of bounds [0, ", num_modes_ - 1, "].");
//
//         int64_t N = x.size(0);
//
//         // --- 1. Base Instance Normalization (without its own affine) ---
//         torch::Tensor x_in; // Instance Normalized x
//         if (x.dim() > 2) { // Input has spatial/sequential dimensions (N, C, D1, ...)
//             std::vector<int64_t> reduce_dims_in;
//             for (int64_t i = 2; i < x.dim(); ++i) reduce_dims_in.push_back(i);
//             auto mean = x.mean(reduce_dims_in, /*keepdim=*/true);
//             auto var = x.var(reduce_dims_in, /*unbiased=*/false, /*keepdim=*/true);
//             x_in = (x - mean) / torch::sqrt(var + eps_in_);
//         } else { // Input is 2D (N, C). InstanceNorm on a single point per channel results in 0.
//             x_in = torch::zeros_like(x);
//         }
//
//         // --- 2. Select mode-specific gamma and beta ---
//         // mode_index could be (N,) or (N,1). We need to select rows from gammas_ and betas_.
//         // gammas_ is (num_modes, num_features)
//         // betas_ is (num_modes, num_features)
//         // torch::index_select or embedding-like lookup
//         torch::Tensor selected_gamma = gammas_.index_select(0, mode_index.view(-1)); // (N_effective, num_features)
//         torch::Tensor selected_beta  = betas_.index_select(0, mode_index.view(-1));  // (N_effective, num_features)
//
//         // N_effective will be N if mode_index is (N,).
//         // If mode_index was a scalar and applied to all N samples, we'd need to handle that.
//         // For simplicity, assume mode_index.view(-1) has N elements.
//         // Or, if mode_index is scalar, repeat it N times.
//         if (mode_index.numel() == 1 && N > 1) {
//             auto single_mode_idx_val = mode_index.item<int64_t>(); // Get the scalar value
//             torch::Tensor single_mode_idx_tensor = torch::tensor({single_mode_idx_val}, torch::kLong).to(mode_index.device());
//             selected_gamma = gammas_.index_select(0, single_mode_idx_tensor); // (1, num_features)
//             selected_beta  = betas_.index_select(0, single_mode_idx_tensor);  // (1, num_features)
//             // These will then broadcast to (N, num_features) when reshaped for affine.
//         } else if (mode_index.numel() != N) {
//              TORCH_CHECK(false, "mode_index numel (", mode_index.numel(), ") must match batch size N (", N, ") or be 1.");
//         }
//
//
//         // --- 3. Reshape selected Gamma and Beta for broadcasting ---
//         // Desired shape: (N, C, 1, 1, ...) to match x_in (N, C, D1, D2, ...)
//         std::vector<int64_t> affine_param_view_shape;
//         affine_param_view_shape.push_back(selected_gamma.size(0)); // Should be N or 1
//         affine_param_view_shape.push_back(num_features_);   // C
//         for (int64_t i = 2; i < x.dim(); ++i) {
//             affine_param_view_shape.push_back(1);
//         }
//
//         selected_gamma = selected_gamma.view(affine_param_view_shape);
//         selected_beta  = selected_beta.view(affine_param_view_shape);
//
//         // --- 4. Apply mode-specific affine transformation ---
//         return selected_gamma * x_in + selected_beta;
//     }
//
//     void pretty_print(std::ostream& stream) const override {
//         stream << "ModeNormalization(num_features=" << num_features_
//                << ", num_modes=" << num_modes_
//                << ", eps_in=" << eps_in_ << ")";
//     }
// };
// TORCH_MODULE(ModeNormalization);
//
//
// // --- Example Usage ---
// int main() {
//     torch::manual_seed(0);
//
//     int64_t num_features = 32;
//     int64_t num_modes = 3;
//     int64_t N = 4; // Batch size
//
//     // --- Test Case 1: 4D input x (NCHW) ---
//     std::cout << "--- Test Case 1: 4D input (NCHW) ---" << std::endl;
//     int64_t H = 8, W = 8;
//     ModeNormalization modenorm_module1(num_features, num_modes);
//     // std::cout << modenorm_module1 << std::endl;
//
//     torch::Tensor x1 = torch::randn({N, num_features, H, W});
//     // Create mode indices for each sample in the batch
//     torch::Tensor mode_indices1 = torch::randint(0, num_modes, {N}, torch::kLong);
//     std::cout << "Input x1 shape: " << x1.sizes() << std::endl;
//     std::cout << "Mode indices1: " << mode_indices1 << std::endl;
//
//     // Modify some gamma/beta for specific modes to see differences
//     modenorm_module1->gammas_.data()[0].fill_(1.5); // Mode 0 gamma = 1.5
//     modenorm_module1->betas_.data()[0].fill_(0.5);  // Mode 0 beta = 0.5
//     modenorm_module1->gammas_.data()[1].fill_(0.8); // Mode 1 gamma = 0.8
//     modenorm_module1->betas_.data()[1].fill_(-0.2); // Mode 1 beta = -0.2
//
//     torch::Tensor y1 = modenorm_module1->forward(x1, mode_indices1);
//     std::cout << "Output y1 shape: " << y1.sizes() << std::endl;
//
//     // Check outputs for different modes
//     for (int64_t i = 0; i < N; ++i) {
//         int64_t current_mode = mode_indices1[i].item<int64_t>();
//         std::cout << "Sample " << i << " (mode " << current_mode << "): output mean "
//                   << y1[i].mean().item<double>() << ", output std " << y1[i].std(false).item<double>() << std::endl;
//         // Expected mean ~ beta_mode, std ~ gamma_mode for that mode
//     }
//     TORCH_CHECK(y1.sizes() == x1.sizes(), "Output y1 shape mismatch!");
//
//
//     // --- Test Case 2: Using a single mode for all batch samples ---
//     std::cout << "\n--- Test Case 2: Single mode for all samples ---" << std::endl;
//     torch::Tensor single_mode_idx = torch::tensor({0}, torch::kLong); // All samples use mode 0
//     std::cout << "Single mode index: " << single_mode_idx << std::endl;
//
//     torch::Tensor y2 = modenorm_module1->forward(x1, single_mode_idx);
//     std::cout << "Output y2 shape: " << y2.sizes() << std::endl;
//     // All samples should now reflect mode 0's gamma/beta characteristics
//     std::cout << "Sample 0 (mode 0 via single_mode_idx): output mean "
//               << y2[0].mean().item<double>() << " (expected ~0.5)" << std::endl;
//     std::cout << "Sample 0 (mode 0 via single_mode_idx): output std "
//               << y2[0].std(false).item<double>() << " (expected ~1.5)" << std::endl;
//     TORCH_CHECK(y2.sizes() == x1.sizes(), "Output y2 shape mismatch!");
//
//
//     // --- Test Case 3: 2D input x (NC) ---
//     std::cout << "\n--- Test Case 3: 2D input (NC) ---" << std::endl;
//     ModeNormalization modenorm_module_2d(num_features, num_modes);
//     modenorm_module_2d->gammas_.data()[0].fill_(2.0);
//     modenorm_module_2d->betas_.data()[0].fill_(1.0);
//
//     torch::Tensor x_2d = torch::randn({N, num_features});
//     torch::Tensor mode_indices_2d = torch::zeros({N}, torch::kLong); // All use mode 0
//     std::cout << "Input x_2d shape: " << x_2d.sizes() << std::endl;
//
//     torch::Tensor y_2d = modenorm_module_2d->forward(x_2d, mode_indices_2d);
//     std::cout << "Output y_2d shape: " << y_2d.sizes() << std::endl;
//     // For 2D input, x_in is 0. So y_2d should be selected_beta.
//     std::cout << "y_2d[0] (mode 0, should be beta_mode0 ~1.0): " << y_2d[0] << std::endl;
//     TORCH_CHECK(torch::allclose(y_2d.select(1,0), torch::tensor(1.0)), "2D output mismatch for beta.");
//
//
//     // --- Test Case 4: Check backward pass ---
//     std::cout << "\n--- Test Case 4: Backward pass check ---" << std::endl;
//     ModeNormalization modenorm_module3(num_features, num_modes);
//     modenorm_module3->train();
//
//     torch::Tensor x3 = torch::randn({N, num_features, H, W}, torch::requires_grad());
//     torch::Tensor mode_indices3 = torch::randint(0, num_modes, {N}, torch::kLong);
//
//     torch::Tensor y3 = modenorm_module3->forward(x3, mode_indices3);
//     torch::Tensor loss = y3.mean();
//     loss.backward();
//
//     bool grad_exists_x3 = x3.grad().defined() && x3.grad().abs().sum().item<double>() > 0;
//     bool grad_exists_gammas = modenorm_module3->gammas_.grad().defined() &&
//                               modenorm_module3->gammas_.grad().abs().sum().item<double>() > 0;
//     bool grad_exists_betas = modenorm_module3->betas_.grad().defined() &&
//                              modenorm_module3->betas_.grad().abs().sum().item<double>() > 0;
//
//     std::cout << "Gradient exists for x3: " << (grad_exists_x3 ? "true" : "false") << std::endl;
//     std::cout << "Gradient exists for gammas: " << (grad_exists_gammas ? "true" : "false") << std::endl;
//     std::cout << "Gradient exists for betas: " << (grad_exists_betas ? "true" : "false") << std::endl;
//
//     TORCH_CHECK(grad_exists_x3, "No gradient for x3!");
//     TORCH_CHECK(grad_exists_gammas, "No gradient for gammas parameters that were used!");
//     TORCH_CHECK(grad_exists_betas, "No gradient for betas parameters that were used!");
//
//
//     std::cout << "\nModeNormalization tests finished." << std::endl;
//     return 0;
// }


namespace xt::norm
{
    ModeNorm::ModeNorm(int64_t num_features, int64_t num_modes, double eps_in)
        : num_features_(num_features),
          num_modes_(num_modes),
          eps_in_(eps_in)
    {
        TORCH_CHECK(num_features > 0, "num_features must be positive.");
        TORCH_CHECK(num_modes > 0, "num_modes must be positive.");

        // Initialize mode-specific learnable affine parameters
        // gammas are typically initialized to 1, betas to 0 for each mode.
        gammas_ = register_parameter("gammas", torch::ones({num_modes_, num_features_}));
        betas_ = register_parameter("betas", torch::zeros({num_modes_, num_features_}));
    }

    auto ModeNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        vector<std::any> tensors_ = tensors;
        auto x = std::any_cast<torch::Tensor>(tensors_[0]);
        auto mode_index = std::any_cast<torch::Tensor>(tensors_[1]);

        // x: input tensor (N, C, D1, D2, ...) where C is num_features_
        // mode_index: LongTensor of shape (N,) or (N, 1) or scalar if broadcasting.
        //             Contains integers from 0 to num_modes_ - 1.
        //             If shape is (N,), each sample in the batch can have a different mode.

        TORCH_CHECK(x.dim() >= 2, "Input tensor x must have at least 2 dimensions (N, C, ...). Got shape ", x.sizes());
        TORCH_CHECK(x.size(1) == num_features_,
                    "Input channels mismatch. Expected ", num_features_, ", got ", x.size(1));
        TORCH_CHECK(mode_index.scalar_type() == torch::kLong, "mode_index must be a LongTensor.");
        TORCH_CHECK(mode_index.min().item<int64_t>() >= 0 && mode_index.max().item<int64_t>() < num_modes_,
                    "mode_index values out of bounds [0, ", num_modes_ - 1, "].");

        int64_t N = x.size(0);

        // --- 1. Base Instance Normalization (without its own affine) ---
        torch::Tensor x_in; // Instance Normalized x
        if (x.dim() > 2)
        {
            // Input has spatial/sequential dimensions (N, C, D1, ...)
            std::vector<int64_t> reduce_dims_in;
            for (int64_t i = 2; i < x.dim(); ++i) reduce_dims_in.push_back(i);
            auto mean = x.mean(reduce_dims_in, /*keepdim=*/true);
            auto var = x.var(reduce_dims_in, /*unbiased=*/false, /*keepdim=*/true);
            x_in = (x - mean) / torch::sqrt(var + eps_in_);
        }
        else
        {
            // Input is 2D (N, C). InstanceNorm on a single point per channel results in 0.
            x_in = torch::zeros_like(x);
        }

        // --- 2. Select mode-specific gamma and beta ---
        // mode_index could be (N,) or (N,1). We need to select rows from gammas_ and betas_.
        // gammas_ is (num_modes, num_features)
        // betas_ is (num_modes, num_features)
        // torch::index_select or embedding-like lookup
        torch::Tensor selected_gamma = gammas_.index_select(0, mode_index.view(-1)); // (N_effective, num_features)
        torch::Tensor selected_beta = betas_.index_select(0, mode_index.view(-1)); // (N_effective, num_features)

        // N_effective will be N if mode_index is (N,).
        // If mode_index was a scalar and applied to all N samples, we'd need to handle that.
        // For simplicity, assume mode_index.view(-1) has N elements.
        // Or, if mode_index is scalar, repeat it N times.
        if (mode_index.numel() == 1 && N > 1)
        {
            auto single_mode_idx_val = mode_index.item<int64_t>(); // Get the scalar value
            torch::Tensor single_mode_idx_tensor = torch::tensor({single_mode_idx_val}, torch::kLong).to(
                mode_index.device());
            selected_gamma = gammas_.index_select(0, single_mode_idx_tensor); // (1, num_features)
            selected_beta = betas_.index_select(0, single_mode_idx_tensor); // (1, num_features)
            // These will then broadcast to (N, num_features) when reshaped for affine.
        }
        else if (mode_index.numel() != N)
        {
            TORCH_CHECK(false, "mode_index numel (", mode_index.numel(), ") must match batch size N (", N,
                        ") or be 1.");
        }


        // --- 3. Reshape selected Gamma and Beta for broadcasting ---
        // Desired shape: (N, C, 1, 1, ...) to match x_in (N, C, D1, D2, ...)
        std::vector<int64_t> affine_param_view_shape;
        affine_param_view_shape.push_back(selected_gamma.size(0)); // Should be N or 1
        affine_param_view_shape.push_back(num_features_); // C
        for (int64_t i = 2; i < x.dim(); ++i)
        {
            affine_param_view_shape.push_back(1);
        }

        selected_gamma = selected_gamma.view(affine_param_view_shape);
        selected_beta = selected_beta.view(affine_param_view_shape);

        // --- 4. Apply mode-specific affine transformation ---
        return selected_gamma * x_in + selected_beta;
    }
}
