// #include "include/normalizations/spectral_normalization.h"
//
//
// //
// // #include <torch/torch.h>
// // #include <iostream>
// // #include <vector>
// // #include <memory> // For std::shared_ptr
// //
// // // Forward declaration for the Impl struct
// // struct SpectralNormWrapperImpl;
// //
// // // The main module struct that users will interact with.
// // // This module wraps another module and applies spectral normalization to its weights.
// // struct SpectralNormWrapper : torch::nn::ModuleHolder<SpectralNormWrapperImpl> {
// //     // Constructor takes the module to wrap and spectral normalization parameters
// //     SpectralNormWrapper(torch::nn::Module module_to_wrap,
// //                         std::string weight_name = "weight", // Name of the weight parameter in the wrapped module
// //                         int64_t num_power_iterations = 1,
// //                         double eps = 1e-12);
// //
// //     // Forward pass will call the wrapped module's forward after applying spectral norm
// //     torch::Tensor forward(torch::Tensor x) {
// //         return impl_->forward(x);
// //     }
// //      // Overload for layers that might take multiple inputs (less common for SN target layers)
// //     template<typename... Args>
// //     torch::Tensor forward(Args&&... args) {
// //         return impl_->forward(std::forward<Args>(args)...);
// //     }
// //
// //     // Method to remove spectral normalization (optional)
// //     void remove_spectral_norm();
// // };
// //
// // // The implementation struct for SpectralNormWrapper
// // struct SpectralNormWrapperImpl : torch::nn::Module {
// //     torch::nn::Module module_; // The module being wrapped (e.g., Linear, Conv2d)
// //     std::string weight_name_;  // Name of the weight parameter (e.g., "weight")
// //     int64_t num_power_iterations_;
// //     double eps_;
// //
// //     // 'u' vector for power iteration, registered as a buffer
// //     // Its shape depends on the weight matrix. Initialized in the constructor or first forward.
// //     torch::Tensor u_;
// //
// //     // Store the original weight parameter before spectral normalization
// //     // This is not strictly needed if we recompute W_sn = W_orig / sigma on every forward pass.
// //     // PyTorch's implementation often stores W_orig and computes W on the fly.
// //     // For simplicity, we'll assume we operate on the module's 'weight' directly for normalization,
// //     // but ideally, we'd have W_orig and W (W being W_orig / sigma).
// //     // Let's try to follow PyTorch's nn.utils.spectral_norm approach:
// //     // It redefines the 'weight' parameter as a computed property.
// //     // We'll store the original weight as 'weight_orig' and compute 'weight' (spectrally_normalized_weight).
// //
// // public:
// //     SpectralNormWrapperImpl(torch::nn::Module module_to_wrap,
// //                             std::string weight_name = "weight",
// //                             int64_t num_power_iterations = 1,
// //                             double eps = 1e-12)
// //         : module_(std::move(module_to_wrap)),
// //           weight_name_(std::move(weight_name)),
// //           num_power_iterations_(num_power_iterations),
// //           eps_(eps) {
// //
// //         TORCH_CHECK(module_ != nullptr, "Module to wrap cannot be null.");
// //         TORCH_CHECK(num_power_iterations_ >= 0, "num_power_iterations must be non-negative.");
// //
// //         // Register the wrapped module so its parameters are part of this module's parameters
// //         // (except for the weight we are spectrally normalizing, which we'll handle).
// //         // Actually, it's better if SpectralNormWrapper *owns* the module.
// //         register_module("module", module_);
// //
// //         // Get the original weight parameter
// //         torch::Tensor weight_orig;
// //         bool found = false;
// //         for (auto& param : module_->named_parameters()) {
// //             if (param.key() == weight_name_) {
// //                 weight_orig = param.value();
// //                 found = true;
// //                 break;
// //             }
// //         }
// //         TORCH_CHECK(found, "Weight parameter '", weight_name_, "' not found in the wrapped module.");
// //
// //         // Store the original weight and remove it from module's direct parameters
// //         // to replace it with a spectrally normalized version. This is complex to do perfectly
// //         // like PyTorch's hook-based approach without deeper C++ API.
// //         // A simpler way for a module wrapper:
// //         // We will compute the spectrally normalized weight on each forward pass and temporarily set it.
// //         // OR, we register 'weight_orig' here and make 'weight' a buffer that gets updated.
// //
// //         // For power iteration, 'u' needs to be initialized.
// //         // Shape of u: if W is (out_features, in_features, ...), u is (out_features) or (in_features)
// //         // Let W be of shape (dim_0, dim_1, ...). Power iteration works on W reshaped to 2D.
// //         // If W is (C_out, C_in, K, K) for Conv2d, reshape to (C_out, C_in*K*K).
// //         // u has shape (dim_0) if W is (dim_0, dim_1)
// //         long M; // Typically out_features
// //         if (weight_orig.dim() == 0) { // Should not happen for Linear/Conv
// //             M = 1;
// //         } else {
// //             M = weight_orig.size(0);
// //         }
// //         u_ = register_buffer("u_vector", torch::randn({M}).to(weight_orig.device()));
// //         // Normalize u initially
// //         u_ = u_ / (u_.norm() + eps_);
// //     }
// //
// //     // Helper to compute spectral norm and normalized weight
// //     static torch::Tensor compute_weight_sn(
// //         const torch::Tensor& weight_orig, // The original weight matrix
// //         torch::Tensor& u,                 // Power iteration vector (will be updated)
// //         int64_t num_power_iterations,
// //         double eps,
// //         bool training) {
// //
// //         // Reshape weight for 2D matrix operations if it's > 2D (e.g., Conv kernels)
// //         auto W_reshaped = weight_orig.reshape({weight_orig.size(0), -1}); // (out_features, other_dims_prod)
// //
// //         torch::Tensor v_hat, u_hat;
// //
// //         if (num_power_iterations == 0) { // Just compute sigma once without iteration (faster but less accurate)
// //             // This is not standard power iteration, but a fallback if num_iter is 0.
// //             // A direct SVD would be too slow for every forward pass.
// //             // If num_power_iterations == 0, PyTorch's spectral_norm uses u and v from the previous step,
// //             // essentially freezing them. Let's adopt this: no update to u, just use it.
// //             v_hat = torch::mv(W_reshaped.t(), u); // W^T * u
// //             v_hat = v_hat / (v_hat.norm() + eps);
// //             u_hat = torch::mv(W_reshaped, v_hat); // W * v
// //             // u is not updated here if num_power_iterations is 0
// //         } else {
// //             // Power iteration for num_power_iterations
// //             // No gradients through power iteration if training, as u is a buffer.
// //             // It is updated in-place.
// //             torch::NoGradGuard no_grad_guard; // Ensure power iteration does not affect gradients
// //             for (int i = 0; i < num_power_iterations; ++i) {
// //                 // v = W^T * u
// //                 v_hat = torch::mv(W_reshaped.t(), u);
// //                 v_hat = v_hat / (v_hat.norm() + eps); // Normalize v
// //                 // u = W * v
// //                 u_hat = torch::mv(W_reshaped, v_hat);
// //                 u = u_hat / (u_hat.norm() + eps); // Normalize u and update in-place
// //             }
// //             // After loop, u is the updated singular vector.
// //             // Recompute v_hat with final u
// //             v_hat = torch::mv(W_reshaped.t(), u);
// //             v_hat = v_hat / (v_hat.norm() + eps);
// //             u_hat = torch::mv(W_reshaped, v_hat); // W * v
// //         }
// //
// //
// //         // Spectral norm sigma = u^T * W * v
// //         // W_reshaped is (M, N), u is (M), v_hat is (N)
// //         // sigma = u.dot(W_reshaped @ v_hat)
// //         // Or more simply, u_hat = W*v, so sigma = ||u_hat|| if u was normalized.
// //         // Or, after iteration, sigma = ||Wv|| / ||v|| where v is an estimate of right singular vector.
// //         // PyTorch implementation: sigma = torch.dot(u, torch.mv(W_reshaped, v_hat))
// //         // More robustly after power iteration: sigma is norm of Wv or W^Tu.
// //         // Since u_hat = W_reshaped @ v_hat, and v_hat is unit norm. sigma = ||u_hat||_2
// //         // Or, if u is updated last: u = Wv / ||Wv||. Then sigma = u^T W v.
// //         // Let's use sigma = u^T * (W * v) = u.dot(u_hat) if u_hat is not normalized yet, or just norm of u_hat.
// //         // A common way: sigma = || W_reshaped @ v_hat ||_2
// //         // If u is the primary vector being iterated: u_{k+1} = W^T W u_k / norm.
// //         // Then sigma = sqrt(lambda_max(W^T W)). Or sigma = || W u_approx_v1 ||
// //         // From PyTorch's impl:
// //         // sigma = torch.dot(u, torch::mv(W_reshaped, v_hat)) is one way.
// //         // Or, after W_reshaped@v_hat (which is u_hat before normalization):
// //         torch::Tensor Wv = torch::mv(W_reshaped, v_hat);
// //         torch::Tensor sigma = torch::dot(u, Wv); // u is unit norm, Wv is W*v_hat
// //
// //         // Normalize the original weight
// //         torch::Tensor W_sn = weight_orig / (sigma + eps);
// //         return W_sn;
// //     }
// //
// //
// //     template<typename... Args>
// //     torch::Tensor forward(Args&&... args) {
// //         // Get the original weight tensor from the wrapped module
// //         torch::Tensor weight_orig = module_->named_parameters()[weight_name_];
// //
// //         // Compute the spectrally normalized weight
// //         // The `u_` buffer is updated in-place by compute_weight_sn
// //         torch::Tensor current_weight_sn = compute_weight_sn(
// //             weight_orig.data(), // Use .data() if we don't want autograd through W_orig here,
// //                                 // SN grad is w.r.t. W_orig directly.
// //             u_,
// //             is_training() ? num_power_iterations_ : 0, // Fewer/no iterations during eval
// //             eps_,
// //             is_training()
// //         );
// //
// //         // --- This is the tricky part for a C++ wrapper ---
// //         // How to make the wrapped module use `current_weight_sn`?
// //         // Option 1: Temporarily replace the parameter (can be messy with autograd).
// //         // Option 2: If the module's forward can take weight as an argument (not typical for nn.Linear/Conv2d).
// //         // Option 3: Re-implement the wrapped module's forward logic here using current_weight_sn. (Most robust)
// //
// //         // For Option 3 (re-implementing forward):
// //         // This requires knowing the type of `module_`.
// //         if (auto linear = module_->as<torch::nn::Linear>()) {
// //             // Original bias, if it exists
// //             auto bias = linear->bias;
// //             return torch::linear(std::get<0>(std::forward_as_tuple(args...)), current_weight_sn, bias);
// //         } else if (auto conv2d = module_->as<torch::nn::Conv2d>()) {
// //             // For Conv2d, need all its parameters (stride, padding, dilation, groups)
// //             // Assume args... contains the input tensor for conv2d
// //             // This is simplified; a real wrapper would need to handle all conv params.
// //             auto bias = conv2d->bias;
// //             return torch::conv2d(std::get<0>(std::forward_as_tuple(args...)), current_weight_sn, bias,
// //                                  conv2d->options.stride(), conv2d->options.padding(),
// //                                  conv2d->options.dilation(), conv2d->options.groups());
// //         } else {
// //             TORCH_CHECK(false, "SpectralNormWrapper: Wrapped module type not currently supported for forward re-implementation. "
// //                                "Supported: Linear, Conv2d. Got: ", module_->name());
// //         }
// //         return torch::Tensor(); // Should not reach here
// //     }
// //
// //     void remove_spectral_norm() {
// //         // This would involve restoring the original weight parameter logic,
// //         // effectively undoing what the constructor/wrapper setup did.
// //         // In PyTorch, it removes hooks and re-assigns weight_orig to weight.
// //         // This is complex to do generically in C++ without explicit parameter management.
// //         TORCH_WARN("remove_spectral_norm() is not fully implemented in this C++ example.");
// //         // For a simple version, one might just stop updating 'u' or bypass the normalization,
// //         // but true removal is deeper.
// //     }
// //
// //     void pretty_print(std::ostream& stream) const override {
// //         stream << "SpectralNormWrapper(module=" << module_->name()
// //                << ", weight_name=" << weight_name_
// //                << ", num_power_iter=" << num_power_iterations_
// //                << ", eps=" << eps_ << ")";
// //     }
// // };
// //
// // // Constructor implementation for SpectralNormWrapper
// // SpectralNormWrapper::SpectralNormWrapper(torch::nn::Module module_to_wrap,
// //                                          std::string weight_name,
// //                                          int64_t num_power_iterations,
// //                                          double eps)
// //     : ModuleHolder(std::make_shared<SpectralNormWrapperImpl>(
// //           std::move(module_to_wrap), std::move(weight_name), num_power_iterations, eps)) {}
// //
// // void SpectralNormWrapper::remove_spectral_norm() {
// //     impl_->remove_spectral_norm();
// // }
// //
// //
// // // --- Example Usage ---
// // int main() {
// //     torch::manual_seed(0);
// //
// //     // --- Test Case 1: Spectral Normalization on a Linear layer ---
// //     std::cout << "--- Test Case 1: SpectralNormWrapper on nn::Linear ---" << std::endl;
// //     int64_t in_features = 10, out_features = 5;
// //     torch::nn::Linear linear_layer(in_features, out_features);
// //     // Store original weight norm for comparison
// //     double orig_weight_spectral_norm = torch::linalg::svdvals(linear_layer->weight).max().item<double>();
// //     std::cout << "Original Linear weight spectral norm (approx): " << orig_weight_spectral_norm << std::endl;
// //
// //     SpectralNormWrapper sn_linear(linear_layer, "weight", 5, 1e-12);
// //     // std::cout << sn_linear << std::endl;
// //
// //     torch::Tensor x1 = torch::randn({4, in_features}); // Batch of 4
// //     std::cout << "Input x1 shape: " << x1.sizes() << std::endl;
// //
// //     // Training pass
// //     sn_linear->train();
// //     torch::Tensor y1_train = sn_linear->forward(x1);
// //     std::cout << "Output y1_train shape: " << y1_train.sizes() << std::endl;
// //
// //     // Check spectral norm of the effective weight used
// //     // We need access to the weight that was used in the forward pass.
// //     // The 'weight' parameter of the original linear_layer is NOT modified in place by this wrapper design.
// //     // The wrapper *calculates* a spectrally normalized weight and uses it.
// //     // To verify, we'd ideally get that `current_weight_sn` or re-run the static method.
// //     torch::Tensor effective_weight_train;
// //     {   // Re-compute for verification
// //         auto W_orig = sn_linear->module_->as<torch::nn::Linear>()->weight;
// //         // Need to access 'u_' from impl. This is tricky without making it public or having a getter.
// //         // For testing, let's just check if the output changed, implying a different weight was used.
// //     }
// //
// //
// //     // Evaluation pass
// //     sn_linear->eval(); // Power iterations might be fewer or zero
// //     torch::Tensor y1_eval = sn_linear->forward(x1);
// //     std::cout << "Output y1_eval shape: " << y1_eval.sizes() << std::endl;
// //     // For proper testing, one would compare y1_train and y1_eval if num_power_iterations changes,
// //     // or check the actual spectral norm of the weight being used.
// //
// //
// //     // --- Test Case 2: Backward pass ---
// //     std::cout << "\n--- Test Case 2: Backward pass check ---" << std::endl;
// //     torch::nn::Linear linear_layer2(in_features, out_features);
// //     SpectralNormWrapper sn_linear_train(linear_layer2, "weight", 1);
// //     sn_linear_train->train();
// //
// //     torch::Tensor x2 = torch::randn({2, in_features}, torch::requires_grad());
// //     torch::Tensor y2 = sn_linear_train->forward(x2);
// //     torch::Tensor loss = y2.mean();
// //     loss.backward();
// //
// //     // Check if original weight in linear_layer2 got gradients
// //     bool grad_exists_orig_weight = linear_layer2->weight.grad().defined() &&
// //                                    linear_layer2->weight.grad().abs().sum().item<double>() > 0;
// //     std::cout << "Gradient exists for original weight: " << (grad_exists_orig_weight ? "true" : "false") << std::endl;
// //     TORCH_CHECK(grad_exists_orig_weight, "Original weight did not receive gradient.");
// //
// //     bool grad_exists_x2 = x2.grad().defined() && x2.grad().abs().sum().item<double>() > 0;
// //     std::cout << "Gradient exists for input x2: " << (grad_exists_x2 ? "true" : "false") << std::endl;
// //     TORCH_CHECK(grad_exists_x2, "Input x2 did not receive gradient.");
// //
// //     // The `u_vector` buffer should not have a gradient
// //     bool no_grad_u_vector = !sn_linear_train->impl_->named_buffers()["u_vector"].grad_fn().defined();
// //     std::cout << "No gradient for u_vector buffer: " << (no_grad_u_vector ? "true":"false") << std::endl;
// //     TORCH_CHECK(no_grad_u_vector, "u_vector buffer should not have a gradient!");
// //
// //
// //     std::cout << "\nSpectralNormWrapper tests finished." << std::endl;
// //     std::cout << "Note: This is a simplified wrapper. PyTorch's nn.utils.spectral_norm is more robust via hooks." << std::endl;
// //     return 0;
// // }
//
//
// namespace xt::norm
// {
//     SpectralNorm::SpectralNorm(xt::Module module_to_wrap,
//                                std::string weight_name,
//                                int64_t num_power_iterations,
//                                double eps)
//         : module_(std::move(module_to_wrap)),
//           weight_name_(std::move(weight_name)),
//           num_power_iterations_(num_power_iterations),
//           eps_(eps)
//     {
//         TORCH_CHECK(module_ != nullptr, "Module to wrap cannot be null.");
//         TORCH_CHECK(num_power_iterations_ >= 0, "num_power_iterations must be non-negative.");
//
//         // Register the wrapped module so its parameters are part of this module's parameters
//         // (except for the weight we are spectrally normalizing, which we'll handle).
//         // Actually, it's better if SpectralNormWrapper *owns* the module.
//         register_module("module", module_);
//
//         // Get the original weight parameter
//         torch::Tensor weight_orig;
//         bool found = false;
//         for (auto& param : module_->named_parameters())
//         {
//             if (param.key() == weight_name_)
//             {
//                 weight_orig = param.value();
//                 found = true;
//                 break;
//             }
//         }
//         TORCH_CHECK(found, "Weight parameter '", weight_name_, "' not found in the wrapped module.");
//
//         // Store the original weight and remove it from module's direct parameters
//         // to replace it with a spectrally normalized version. This is complex to do perfectly
//         // like PyTorch's hook-based approach without deeper C++ API.
//         // A simpler way for a module wrapper:
//         // We will compute the spectrally normalized weight on each forward pass and temporarily set it.
//         // OR, we register 'weight_orig' here and make 'weight' a buffer that gets updated.
//
//         // For power iteration, 'u' needs to be initialized.
//         // Shape of u: if W is (out_features, in_features, ...), u is (out_features) or (in_features)
//         // Let W be of shape (dim_0, dim_1, ...). Power iteration works on W reshaped to 2D.
//         // If W is (C_out, C_in, K, K) for Conv2d, reshape to (C_out, C_in*K*K).
//         // u has shape (dim_0) if W is (dim_0, dim_1)
//         long M; // Typically out_features
//         if (weight_orig.dim() == 0)
//         {
//             // Should not happen for Linear/Conv
//             M = 1;
//         }
//         else
//         {
//             M = weight_orig.size(0);
//         }
//         u_ = register_buffer("u_vector", torch::randn({M}).to(weight_orig.device()));
//         // Normalize u initially
//         u_ = u_ / (u_.norm() + eps_);
//     }
//
//     auto SpectralNorm::forward(std::initializer_list<std::any> tensors) -> std::any
//     {
//         return torch::zeros(10);
//     }
// }
