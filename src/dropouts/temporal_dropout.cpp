#include "include/dropouts/temporal_dropout.h"


// #include <torch/torch.h>
// #include <vector>
// #include <ostream> // For std::ostream

// struct TemporalDropoutImpl : torch::nn::Module {
//     double p_drop_timestep_; // Probability of dropping an entire time step.
//     int time_dim_;           // The dimension index representing time/sequence length.
//     double epsilon_ = 1e-7;
//
//     // By default, assumes time_dim is 1 for (Batch, SeqLen, Features)
//     // or 0 for (SeqLen, Features) or (SeqLen, Batch, Features) if batch_first=false for an RNN.
//     // For simplicity, we'll make it configurable, defaulting to 1 (Batch, SeqLen, Feat).
//     TemporalDropoutImpl(double p_drop_timestep = 0.1, int time_dim = 1)
//         : p_drop_timestep_(p_drop_timestep), time_dim_(time_dim) {
//         TORCH_CHECK(p_drop_timestep_ >= 0.0 && p_drop_timestep_ <= 1.0,
//                     "TemporalDropout p_drop_timestep must be between 0 and 1.");
//         TORCH_CHECK(time_dim_ >= 0, "time_dim must be non-negative.");
//     }
//
//     torch::Tensor forward(const torch::Tensor& input) {
//         if (!this->is_training() || p_drop_timestep_ == 0.0) {
//             return input;
//         }
//         if (p_drop_timestep_ == 1.0) {
//             return torch::zeros_like(input);
//         }
//
//         TORCH_CHECK(input.dim() > time_dim_, "Input tensor must have at least time_dim + 1 dimensions.");
//
//         int64_t seq_len = input.size(time_dim_);
//         if (seq_len == 0) return input; // No time steps to drop
//
//         double keep_prob = 1.0 - p_drop_timestep_;
//
//         // Create a mask for time steps: shape (seq_len)
//         // This mask will decide which time steps are kept (1) or dropped (0).
//         torch::Tensor timestep_mask_1d = torch::bernoulli(
//             torch::full({seq_len}, keep_prob, input.options())
//         ).to(input.dtype());
//
//         // Reshape timestep_mask_1d to be broadcastable with the input tensor.
//         // Example: If input is (Batch, SeqLen, Features) and time_dim=1,
//         // mask should be (1, SeqLen, 1) to broadcast over Batch and Features.
//         // Or (Batch, SeqLen, 1) if we want different time steps dropped per batch item.
//         // Let's implement the simpler version: same time steps dropped for all batch items.
//         std::vector<int64_t> broadcast_mask_shape(input.dim(), 1L);
//         broadcast_mask_shape[time_dim_] = seq_len;
//
//         torch::Tensor broadcastable_mask = timestep_mask_1d.view(broadcast_mask_shape);
//
//         // Apply mask and scale (inverted dropout)
//         // Scaling should be based on the number of time steps actually kept.
//         double num_kept_timesteps = timestep_mask_1d.sum().item<double>();
//         double scale_factor;
//         if (num_kept_timesteps > 0) {
//             // scale_factor = 1.0 / keep_prob; // This is if we expect keep_prob fraction of steps
//             scale_factor = static_cast<double>(seq_len) / (num_kept_timesteps + epsilon_);
//         } else { // All time steps were dropped
//             scale_factor = 0; // Effectively zeros out output, which is correct
//         }
//         // If using the simple 1.0/keep_prob, then the scaling factor is just that:
//         // scale_factor = 1.0 / (keep_prob + epsilon_);
//         // Let's use the actual proportion kept for more accurate scaling like DropBlock/SensorDropout.
//
//         return (input * broadcastable_mask) * scale_factor;
//     }
//
//     void pretty_print(std::ostream& stream) const override {
//         stream << "TemporalDropout(p_drop_timestep=" << p_drop_timestep_
//                << ", time_dim=" << time_dim_ << ")";
//     }
// };
//
// TORCH_MODULE(TemporalDropout);
//
// /*
// // Example of how to use the TemporalDropout module:
// #include <iostream>
// #include <iomanip> // For std::fixed, std::setprecision
//
// void run_temporal_dropout_example() {
//     torch::manual_seed(0);
//     std::cout << std::fixed << std::setprecision(4);
//
//     double prob_step_drop = 0.4; // 40% chance of dropping an entire time step
//
//     // --- Test with 3D input (Batch, SeqLen, Features), time_dim=1 (default) ---
//     TemporalDropout dropout_module_3d(prob_step_drop, 1);
//     std::cout << "TemporalDropout Module (3D, time_dim=1): " << dropout_module_3d << std::endl;
//
//     // Batch=2, SeqLen=5, Features=3
//     torch::Tensor input_3d = torch::ones({2, 5, 3});
//     // Make time steps visually distinct for one batch item for easier checking
//     for (int64_t t = 0; t < input_3d.size(1); ++t) {
//         input_3d.select(0, 0).select(0, t) *= (t + 1); // Batch 0, time t, features *= (t+1)
//         input_3d.select(0, 1).select(0, t) *= (t + 1) * 10; // Batch 1
//     }
//
//     std::cout << "\nInput 3D (B,S,F) (Batch 0, sum over features per step):\n"
//               << input_3d[0].sum(-1) << std::endl;
//
//     dropout_module_3d->train(); // Set to training mode
//     torch::Tensor output_3d_train = dropout_module_3d(input_3d);
//     std::cout << "Output 3D (train) (Batch 0, sum over features per step):\n"
//               << output_3d_train[0].sum(-1) << std::endl;
//     // Expected: Some entire time steps (e.g., all features at t=2) will be zeroed out
//     // across ALL batch items (because the mask is (1, SeqLen, 1)).
//     // Kept time steps will be scaled.
//
//     dropout_module_3d->eval(); // Set to evaluation mode
//     torch::Tensor output_3d_eval = dropout_module_3d(input_3d);
//     std::cout << "Output 3D (eval) (Batch 0, sum over features per step):\n"
//               << output_3d_eval[0].sum(-1) << std::endl;
//     TORCH_CHECK(torch::allclose(input_3d, output_3d_eval), "TemporalDropout 3D eval output mismatch!");
//
//
//     // --- Test with 2D input (SeqLen, Features), time_dim=0 ---
//     TemporalDropout dropout_module_2d(prob_step_drop, 0);
//     std::cout << "\nTemporalDropout Module (2D, time_dim=0): " << dropout_module_2d << std::endl;
//
//     // SeqLen=6, Features=2
//     torch::Tensor input_2d = torch::ones({6, 2});
//     for (int64_t t = 0; t < input_2d.size(0); ++t) {
//         input_2d[t] *= (t + 1);
//     }
//     std::cout << "Input 2D (S,F) (sum over features per step):\n" << input_2d.sum(-1) << std::endl;
//
//     dropout_module_2d->train();
//     torch::Tensor output_2d_train = dropout_module_2d(input_2d);
//     std::cout << "Output 2D (train) (sum over features per step):\n" << output_2d_train.sum(-1) << std::endl;
//
//
//     // --- Test with p_drop_timestep = 0.0 (no dropout) ---
//     TemporalDropout no_drop_module(0.0, 1);
//     no_drop_module->train();
//     torch::Tensor output_no_drop_train = no_drop_module(input_3d);
//     std::cout << "\nOutput 3D (train, p_drop_timestep=0.0) (Batch 0, sum over features per step):\n"
//               << output_no_drop_train[0].sum(-1) << std::endl;
//     TORCH_CHECK(torch::allclose(input_3d, output_no_drop_train), "TemporalDropout p_drop=0.0 output mismatch!");
//
//     // --- Test with p_drop_timestep = 1.0 (drop all steps) ---
//     TemporalDropout full_drop_module(1.0, 1);
//     full_drop_module->train();
//     torch::Tensor output_full_drop_train = full_drop_module(input_3d);
//     std::cout << "\nOutput 3D (train, p_drop_timestep=1.0) (Batch 0, sum over features per step):\n"
//               << output_full_drop_train[0].sum(-1) << std::endl;
//     TORCH_CHECK(torch::allclose(torch::zeros_like(input_3d), output_full_drop_train), "TemporalDropout p_drop=1.0 output mismatch!");
//
// }
//
// // int main() {
// //    run_temporal_dropout_example();
// //    return 0;
// // }
//
// ## REFINEMENT
//
// // // If input is (Batch, SeqLen, Features) and time_dim=1,
// // // and we want different time steps dropped per batch item:
// // // Mask shape should be (Batch, SeqLen, 1)
// // std::vector<int64_t> per_batch_mask_shape = {input.size(0), seq_len};
// // for (int d = 0; d < input.dim(); ++d) {
// //     if (d != 0 && d != time_dim_) { // 0 is batch, time_dim_ is time
// //         per_batch_mask_shape.push_back(1); // Add singleton for other dims like Features
// //     }
// // }
// // This gets more complex. Simpler for (B,S,F) -> mask (B,S,1):
// // torch::Tensor per_batch_timestep_mask = torch::bernoulli(
// //     torch::full({input.size(0), seq_len}, keep_prob, input.options())
// // ).to(input.dtype());
// // broadcastable_mask = per_batch_timestep_mask.unsqueeze(-1); // Add feature dim for broadcasting
//
// // // Scaling would then be per batch item based on how many steps were kept for *that* item.
// // torch::Tensor num_kept_per_batch = per_batch_timestep_mask.sum(/*dim=*/1, /*keepdim=*/true); // Sum over SeqLen
// // torch::Tensor scale_per_batch = static_cast<double>(seq_len) / (num_kept_per_batch + epsilon_);
// // return (input * broadcastable_mask) * scale_per_batch.unsqueeze(-1); // Ensure scale also broadcasts over features
//
//


namespace xt::dropouts
{
    TemporalDropout::TemporalDropout(double p_drop_timestep , int time_dim )
        : p_drop_timestep_(p_drop_timestep), time_dim_(time_dim)
    {
        TORCH_CHECK(p_drop_timestep_ >= 0.0 && p_drop_timestep_ <= 1.0,
                    "TemporalDropout p_drop_timestep must be between 0 and 1.");
        TORCH_CHECK(time_dim_ >= 0, "time_dim must be non-negative.");
    }


    auto TemporalDropout::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        vector<std::any> tensors_ = tensors;
        auto input = std::any_cast<torch::Tensor>(tensors_[0]);


        if (!this->is_training() || p_drop_timestep_ == 0.0)
        {
            return input;
        }
        if (p_drop_timestep_ == 1.0)
        {
            return torch::zeros_like(input);
        }

        TORCH_CHECK(input.dim() > time_dim_, "Input tensor must have at least time_dim + 1 dimensions.");

        int64_t seq_len = input.size(time_dim_);
        if (seq_len == 0) return input; // No time steps to drop

        double keep_prob = 1.0 - p_drop_timestep_;

        // Create a mask for time steps: shape (seq_len)
        // This mask will decide which time steps are kept (1) or dropped (0).
        torch::Tensor timestep_mask_1d = torch::bernoulli(
            torch::full({seq_len}, keep_prob, input.options())
        ).to(input.dtype());

        // Reshape timestep_mask_1d to be broadcastable with the input tensor.
        // Example: If input is (Batch, SeqLen, Features) and time_dim=1,
        // mask should be (1, SeqLen, 1) to broadcast over Batch and Features.
        // Or (Batch, SeqLen, 1) if we want different time steps dropped per batch item.
        // Let's implement the simpler version: same time steps dropped for all batch items.
        std::vector<int64_t> broadcast_mask_shape(input.dim(), 1L);
        broadcast_mask_shape[time_dim_] = seq_len;

        torch::Tensor broadcastable_mask = timestep_mask_1d.view(broadcast_mask_shape);

        // Apply mask and scale (inverted dropout)
        // Scaling should be based on the number of time steps actually kept.
        double num_kept_timesteps = timestep_mask_1d.sum().item<double>();
        double scale_factor;
        if (num_kept_timesteps > 0)
        {
            // scale_factor = 1.0 / keep_prob; // This is if we expect keep_prob fraction of steps
            scale_factor = static_cast<double>(seq_len) / (num_kept_timesteps + epsilon_);
        }
        else
        {
            // All time steps were dropped
            scale_factor = 0; // Effectively zeros out output, which is correct
        }
        // If using the simple 1.0/keep_prob, then the scaling factor is just that:
        // scale_factor = 1.0 / (keep_prob + epsilon_);
        // Let's use the actual proportion kept for more accurate scaling like DropBlock/SensorDropout.

        return (input * broadcastable_mask) * scale_factor;
    }
}
