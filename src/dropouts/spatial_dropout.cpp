#include "include/dropouts/spatial_dropout.h"


//
// #include <torch/torch.h>
// #include <vector>
// #include <ostream> // For std::ostream
//
// struct SpatialDropoutImpl : torch::nn::Module {
//     double p_drop_channel_; // Probability of dropping an entire channel.
//     double epsilon_ = 1e-7;   // For numerical stability
//
//     SpatialDropoutImpl(double p_drop_channel = 0.5) : p_drop_channel_(p_drop_channel) {
//         TORCH_CHECK(p_drop_channel_ >= 0.0 && p_drop_channel_ <= 1.0,
//                     "SpatialDropout probability p_drop_channel must be between 0 and 1.");
//     }
//
//     // Input x is expected to be at least 2D.
//     // For NCHW (e.g., Conv2D output): dim 1 is Channel.
//     // For NCL  (e.g., Conv1D output): dim 1 is Channel.
//     // For NC   (e.g., after flattening spatial dims but keeping channel distinct): dim 1 is Channel.
//     // If input is (Batch, Channels), it means each "channel" is a single feature.
//     torch::Tensor forward(const torch::Tensor& input) {
//         if (!this->is_training() || p_drop_channel_ == 0.0) {
//             return input;
//         }
//         if (p_drop_channel_ == 1.0) {
//             return torch::zeros_like(input);
//         }
//
//         TORCH_CHECK(input.dim() >= 2, "SpatialDropout input must be at least 2D (e.g., [Batch, Channels, ...]).");
//
//         int64_t num_channels = input.size(1); // Assuming channel dimension is 1
//         if (num_channels == 0) return input; // No channels to drop
//
//         double keep_prob = 1.0 - p_drop_channel_;
//
//         // Create a mask for channels: shape (1, num_channels, 1, 1, ...) or (Batch, num_channels, 1, 1,...)
//         // The paper suggests a per-batch-item channel mask, but often a single channel mask
//         // is applied to all items in the batch for simplicity/efficiency, or per-batch-item mask.
//         // Let's implement per-batch-item channel mask for generality.
//         // Mask shape: (BatchSize, NumChannels)
//         torch::Tensor channel_mask_2d = torch::bernoulli(
//             torch::full({input.size(0), num_channels}, keep_prob, input.options())
//         ).to(input.dtype());
//
//         // Reshape channel_mask_2d to be broadcastable with the input tensor.
//         // If input is (N, C, H, W), mask should be (N, C, 1, 1).
//         // If input is (N, C, L), mask should be (N, C, 1).
//         // If input is (N, C), mask is already (N, C).
//         torch::Tensor broadcastable_mask = channel_mask_2d;
//         for (int d = 2; d < input.dim(); ++d) {
//             broadcastable_mask = broadcastable_mask.unsqueeze(-1);
//         }
//         // Now broadcastable_mask has shape like (N, C, 1, ..., 1) matching input's rank.
//
//         // Apply mask and scale (inverted dropout)
//         return (input * broadcastable_mask) / (keep_prob + epsilon_);
//     }
//
//     void pretty_print(std::ostream& stream) const override {
//         stream << "SpatialDropout(p_drop_channel=" << p_drop_channel_ << ")";
//     }
// };
//
// TORCH_MODULE(SpatialDropout); // Creates the SpatialDropout module "class"
//
// /*
// // Example of how to use the SpatialDropout module:
// // (This is for illustration and would typically be in your main application code)
//
// #include <iostream>
//
// void run_spatial_dropout_example() {
//     torch::manual_seed(0); // For reproducible results
//
//     double prob_channel_drop = 0.5; // 50% chance of dropping a whole channel
//     SpatialDropout spatial_dropout_module(prob_channel_drop);
//     std::cout << "SpatialDropout Module: " << spatial_dropout_module << std::endl;
//
//     // --- Test with 4D input (like Conv2D output: NCHW) ---
//     // Batch=2, Channels=3, Height=2, Width=2
//     torch::Tensor input_4d = torch::ones({2, 3, 2, 2});
//     // Make channels distinct for better visualization
//     input_4d.select(1, 0) *= 1; // Channel 0 all 1s
//     input_4d.select(1, 1) *= 2; // Channel 1 all 2s
//     input_4d.select(1, 2) *= 3; // Channel 2 all 3s
//
//     std::cout << "\nInput 4D (NCHW) (Original values):\n" << input_4d << std::endl;
//
//     spatial_dropout_module->train(); // Set to training mode
//     torch::Tensor output_4d_train = spatial_dropout_module->forward(input_4d);
//     std::cout << "Output 4D (train):\n" << output_4d_train << std::endl;
//     // Expected: For each batch item, some entire channels (e.g., all 1s, or all 2s, or all 3s sections)
//     // will be zeroed out. Kept channels will be scaled by 1 / (1 - 0.5) = 2.
//     // So, kept channel 0 would become all 2s, kept channel 1 all 4s, kept channel 2 all 6s.
//
//     spatial_dropout_module->eval(); // Set to evaluation mode
//     torch::Tensor output_4d_eval = spatial_dropout_module->forward(input_4d);
//     std::cout << "Output 4D (eval):\n" << output_4d_eval << std::endl;
//     TORCH_CHECK(torch::allclose(input_4d, output_4d_eval), "SpatialDropout 4D eval output mismatch!");
//
//
//     // --- Test with 3D input (like Conv1D output: NCL) ---
//     // Batch=2, Channels=3, Length=4
//     torch::Tensor input_3d = torch::ones({2, 3, 4});
//     input_3d.select(1, 0) *= 10; // Channel 0
//     input_3d.select(1, 1) *= 20; // Channel 1
//     input_3d.select(1, 2) *= 30; // Channel 2
//     std::cout << "\nInput 3D (NCL) (Original values):\n" << input_3d << std::endl;
//
//     spatial_dropout_module->train();
//     torch::Tensor output_3d_train = spatial_dropout_module->forward(input_3d);
//     std::cout << "Output 3D (train):\n" << output_3d_train << std::endl;
//
//
//     // --- Test with 2D input (Batch, Channels/Features) ---
//     // Batch=3, Channels=4
//     torch::Tensor input_2d = torch::ones({3, 4});
//     input_2d[0] *= 1; input_2d[1] *= 2; input_2d[2] *= 3; // Make batches distinct
//     std::cout << "\nInput 2D (NC) (Original values):\n" << input_2d << std::endl;
//
//     spatial_dropout_module->train();
//     torch::Tensor output_2d_train = spatial_dropout_module->forward(input_2d);
//     std::cout << "Output 2D (train):\n" << output_2d_train << std::endl;
//     // Here, "channels" are just individual features per batch item.
//     // So, for each batch item, some features will be zeroed, others scaled.
// }
//
// // int main() {
// //    run_spatial_dropout_example();
// //    return 0;
// // }
// */
//


namespace xt::dropouts
{
    SpatialDropout::SpatialDropout(double p_drop_channel) : p_drop_channel_(p_drop_channel)
    {
        TORCH_CHECK(p_drop_channel_ >= 0.0 && p_drop_channel_ <= 1.0,
                    "SpatialDropout probability p_drop_channel must be between 0 and 1.");
    }


    auto SpatialDropout::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        vector<std::any> tensors_ = tensors;
        auto input = std::any_cast<torch::Tensor>(tensors_[0]);

        if (!this->is_training() || p_drop_channel_ == 0.0)
        {
            return input;
        }
        if (p_drop_channel_ == 1.0)
        {
            return torch::zeros_like(input);
        }

        TORCH_CHECK(input.dim() >= 2, "SpatialDropout input must be at least 2D (e.g., [Batch, Channels, ...]).");

        int64_t num_channels = input.size(1); // Assuming channel dimension is 1
        if (num_channels == 0) return input; // No channels to drop

        double keep_prob = 1.0 - p_drop_channel_;

        // Create a mask for channels: shape (1, num_channels, 1, 1, ...) or (Batch, num_channels, 1, 1,...)
        // The paper suggests a per-batch-item channel mask, but often a single channel mask
        // is applied to all items in the batch for simplicity/efficiency, or per-batch-item mask.
        // Let's implement per-batch-item channel mask for generality.
        // Mask shape: (BatchSize, NumChannels)
        torch::Tensor channel_mask_2d = torch::bernoulli(
            torch::full({input.size(0), num_channels}, keep_prob, input.options())
        ).to(input.dtype());

        // Reshape channel_mask_2d to be broadcastable with the input tensor.
        // If input is (N, C, H, W), mask should be (N, C, 1, 1).
        // If input is (N, C, L), mask should be (N, C, 1).
        // If input is (N, C), mask is already (N, C).
        torch::Tensor broadcastable_mask = channel_mask_2d;
        for (int d = 2; d < input.dim(); ++d)
        {
            broadcastable_mask = broadcastable_mask.unsqueeze(-1);
        }
        // Now broadcastable_mask has shape like (N, C, 1, ..., 1) matching input's rank.

        // Apply mask and scale (inverted dropout)
        return (input * broadcastable_mask) / (keep_prob + epsilon_);
    }
}
