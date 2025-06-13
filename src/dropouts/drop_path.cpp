#include "include/dropouts/drop_path.h"

// #include <torch/torch.h>
// #include <vector>
// #include <ostream> // For std::ostream
//
// struct DropPathImpl : torch::nn::Module {
//     double p_drop_;
//     double epsilon_ = 1e-7; // For numerical stability in division
//
//     DropPathImpl(double p_drop = 0.1) : p_drop_(p_drop) {
//         TORCH_CHECK(p_drop_ >= 0.0 && p_drop_ <= 1.0, "DropPath probability p_drop must be between 0 and 1.");
//     }
//
//     torch::Tensor forward(const torch::Tensor& input) {
//         if (!this->is_training() || p_drop_ == 0.0) {
//             return input;
//         }
//         if (p_drop_ == 1.0) {
//             return torch::zeros_like(input);
//         }
//
//         TORCH_CHECK(input.dim() >= 1, "DropPath input must have at least one dimension (expected batch dimension at dim 0).");
//
//         int64_t batch_size = input.size(0);
//         double keep_prob = 1.0 - p_drop_;
//
//         // Create a per-sample mask (1 to keep, 0 to drop)
//         torch::Tensor random_tensor = torch::rand({batch_size}, input.options());
//         torch::Tensor keep_mask_1d = (random_tensor < keep_prob).to(input.dtype());
//
//         // Reshape mask to be broadcastable with input: (N, 1, 1, ...)
//         std::vector<int64_t> view_shape(input.dim(), 1L);
//         if (input.dim() > 0) { // Should always be true due to TORCH_CHECK above
//             view_shape[0] = batch_size;
//         } else { // Should not happen, but defensive
//              return input; // Or error
//         }
//         torch::Tensor keep_mask = keep_mask_1d.view(view_shape);
//
//         // Apply mask and scale (inverted dropout)
//         return (input * keep_mask) / (keep_prob + epsilon_);
//     }
//
//     void pretty_print(std::ostream& stream) const override {
//         stream << "DropPath(p_drop=" << p_drop_ << ")";
//     }
// };
//
// TORCH_MODULE(DropPath); // Creates the DropPath module "class"
//
// /*
// // Example of how to use the DropPath module:
// // (This is for illustration and would typically be in your main application code)
//
// #include <iostream>
//
// void run_drop_path_example() {
//     torch::manual_seed(1); // For reproducible results
//
//     double drop_probability = 0.5;
//     DropPath drop_path_module(drop_probability);
//     std::cout << "DropPath Module: " << drop_path_module << std::endl;
//
//     // Example input tensor (Batch, Channels, Height, Width)
//     // Let's use a small batch size to clearly see per-sample dropping
//     torch::Tensor input_tensor = torch::ones({4, 2, 2, 2}); // Batch=4, C=2, H=2, W=2
//     input_tensor[1] *= 2.0; // Make samples distinct
//     input_tensor[2] *= 3.0;
//     input_tensor[3] *= 4.0;
//
//     std::cout << "Input Tensor (shape " << input_tensor.sizes() << "):" << std::endl;
//     // For brevity, let's just print sum per sample
//     for (int i=0; i < input_tensor.size(0); ++i) {
//         std::cout << "Input sample " << i << " sum: " << input_tensor[i].sum().item<float>() << std::endl;
//     }
//
//
//     // --- Training mode ---
//     drop_path_module->train(); // Set the module to training mode
//     torch::Tensor output_train = drop_path_module->forward(input_tensor);
//     std::cout << "\nOutput (training mode, p_drop=" << drop_probability << "):" << std::endl;
//     for (int i=0; i < output_train.size(0); ++i) {
//         std::cout << "Output sample " << i << " sum: " << output_train[i].sum().item<float>()
//                   << (output_train[i].sum().item<float>() == 0 ? " (Dropped)" : " (Kept & Scaled)")
//                   << std::endl;
//     }
//     // Expected: Approx 50% of samples (entire BxCxHxW slice) will be zeroed out.
//     // Non-zero samples will be scaled by 1 / (1 - 0.5) = 2.
//     // So original sums of 8, 16, 24, 32 would become 0 or (16, 32, 48, 64) respectively.
//
//     // --- Evaluation mode ---
//     drop_path_module->eval(); // Set the module to evaluation mode
//     torch::Tensor output_eval = drop_path_module->forward(input_tensor);
//     std::cout << "\nOutput (evaluation mode):" << std::endl;
//      for (int i=0; i < output_eval.size(0); ++i) {
//         std::cout << "Output sample " << i << " sum: " << output_eval[i].sum().item<float>() << std::endl;
//     }
//     // Expected output to be identical to input in evaluation mode.
//     TORCH_CHECK(torch::allclose(input_tensor, output_eval), "DropPath eval output mismatch!");
//
//
//     // --- Example with p_drop = 0.0 (no dropout) ---
//     DropPath no_drop_module(0.0);
//     no_drop_module->train();
//     torch::Tensor output_no_drop = no_drop_module->forward(input_tensor);
//     std::cout << "\nOutput (training mode, p_drop=0.0):" << std::endl;
//     for (int i=0; i < output_no_drop.size(0); ++i) {
//         std::cout << "Output sample " << i << " sum: " << output_no_drop[i].sum().item<float>() << std::endl;
//     }
//     TORCH_CHECK(torch::allclose(input_tensor, output_no_drop), "DropPath p_drop=0.0 output mismatch!");
// }
//
// // int main() {
// //    run_drop_path_example();
// //    return 0;
// // }
// */


namespace xt::dropouts
{
    DropPath::DropPath(double p_drop) : p_drop_(p_drop)
    {
        TORCH_CHECK(p_drop_ >= 0.0 && p_drop_ <= 1.0, "DropPath probability p_drop must be between 0 and 1.");
    }

    auto DropPath::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        vector<std::any> tensors_ = tensors;
        auto input = std::any_cast<torch::Tensor>(tensors_[0]);

        if (!this->is_training() || p_drop_ == 0.0)
        {
            return input;
        }
        if (p_drop_ == 1.0)
        {
            return torch::zeros_like(input);
        }

        TORCH_CHECK(input.dim() >= 1,
                    "DropPath input must have at least one dimension (expected batch dimension at dim 0).");

        int64_t batch_size = input.size(0);
        double keep_prob = 1.0 - p_drop_;

        // Create a per-sample mask (1 to keep, 0 to drop)
        torch::Tensor random_tensor = torch::rand({batch_size}, input.options());
        torch::Tensor keep_mask_1d = (random_tensor < keep_prob).to(input.dtype());

        // Reshape mask to be broadcastable with input: (N, 1, 1, ...)
        std::vector<int64_t> view_shape(input.dim(), 1L);
        if (input.dim() > 0)
        {
            // Should always be true due to TORCH_CHECK above
            view_shape[0] = batch_size;
        }
        else
        {
            // Should not happen, but defensive
            return input; // Or error
        }
        torch::Tensor keep_mask = keep_mask_1d.view(view_shape);

        // Apply mask and scale (inverted dropout)
        return (input * keep_mask) / (keep_prob + epsilon_);
    }
}
