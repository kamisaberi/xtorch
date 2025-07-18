#include <dropouts/shake_drop.h>


// #include <torch/torch.h>
// #include <random>  // For std::uniform_real_distribution, std::mt19937
// #include <ostream> // For std::ostream
//
// struct ShakeDropImpl : torch::nn::Module {
//     double p_drop_;      // Probability of applying alpha scaling (the "drop" or "shake" event)
//     double alpha_range_min_ = -1.0;
//     double alpha_range_max_ = 1.0;
//     double beta_range_min_  = 0.0; // Corresponds to 1-c with c=1 from paper
//     double beta_range_max_  = 2.0; // Corresponds to 1+c with c=1 from paper
//
//     // For random number generation (per-instance, could be static for global seed)
//     std::mt19937 gen_;
//
//
//     ShakeDropImpl(double p_drop = 0.5)
//         : p_drop_(p_drop), gen_(std::random_device{}()) { // Seed with random_device
//         TORCH_CHECK(p_drop_ >= 0.0 && p_drop_ <= 1.0, "p_drop must be between 0 and 1.");
//     }
//
//     // Input is the output of the residual branch
//     torch::Tensor forward(const torch::Tensor& branch_output) {
//         if (!this->is_training()) {
//             // Evaluation mode: scale by expected value of the forward scaling factor.
//             // E[scaling_factor] = p_drop_ * E[alpha] + (1 - p_drop_) * E[beta]
//             // E[alpha] = (alpha_range_min_ + alpha_range_max_) / 2.0 = (-1+1)/2 = 0
//             // E[beta]  = (beta_range_min_ + beta_range_max_) / 2.0   = (0+2)/2 = 1
//             // So, E[scaling_factor] = p_drop_ * 0 + (1 - p_drop_) * 1 = 1 - p_drop_
//             return branch_output * (1.0 - p_drop_);
//         }
//
//         // Training mode: apply stochastic scaling
//         double rand_val_for_choice = std::uniform_real_distribution<double>(0.0, 1.0)(gen_);
//         double scale_factor;
//
//         if (rand_val_for_choice < p_drop_) {
//             // Apply alpha scaling (uniform from [-1, 1])
//             scale_factor = std::uniform_real_distribution<double>(
//                 alpha_range_min_, alpha_range_max_
//             )(gen_);
//         } else {
//             // Apply beta scaling (uniform from [0, 2])
//             scale_factor = std::uniform_real_distribution<double>(
//                 beta_range_min_, beta_range_max_
//             )(gen_);
//         }
//
//         // The scale_factor is a scalar, applied to the entire branch_output tensor.
//         // The paper suggests per-channel or per-sample randomness for some variants (Shake-Shake),
//         // but the core ShakeDrop applies a single random scalar per branch activation.
//         return branch_output * scale_factor;
//     }
//
//     void pretty_print(std::ostream& stream) const override {
//         stream << "ShakeDrop(p_drop=" << p_drop_
//                << ", alpha_range=[" << alpha_range_min_ << "," << alpha_range_max_ << "]"
//                << ", beta_range=[" << beta_range_min_ << "," << beta_range_max_ << "]"
//                << ")";
//     }
// };
//
// TORCH_MODULE(ShakeDrop);
//
//
// /*
// // --- Example: How ShakeDrop might be used in a residual block ---
//
// struct ResidualBlockWithShakeDrop : torch::nn::Module {
//     torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
//     ShakeDrop shake_drop_module; // Instance of ShakeDrop
//
//     ResidualBlockWithShakeDrop(int channels, double sd_p_drop = 0.5)
//         : shake_drop_module(sd_p_drop) {
//         conv1 = register_module("conv1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(channels, channels, 3).padding(1)));
//         conv2 = register_module("conv2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(channels, channels, 3).padding(1)));
//         // shake_drop_module is already initialized.
//         // If it had learnable parameters, it would need to be registered.
//         // Since it doesn't, direct member usage is fine.
//     }
//
//     torch::Tensor forward(const torch::Tensor& x) {
//         torch::Tensor identity = x;
//         torch::Tensor branch = torch::relu(conv1(x));
//         branch = conv2(branch); // Output of the residual branch
//
//         // Apply ShakeDrop to the branch output
//         // The ShakeDrop module handles training/eval mode internally.
//         branch = shake_drop_module(branch);
//
//         return torch::relu(identity + branch);
//     }
// };
// TORCH_MODULE(ResidualBlockWithShakeDrop);
//
//
// #include <iostream>
// #include <iomanip> // For std::fixed, std::setprecision
//
// void run_shake_drop_example() {
//     // For ShakeDrop, seeding the module's generator directly is more complex.
//     // We rely on std::random_device for initial seeding within the module.
//     // For full reproducibility here, one might pass a seed to ShakeDropImpl constructor.
//     // torch::manual_seed(0); // Seeds PyTorch's global generator, not std::mt19937 in ShakeDropImpl
//
//     double p_drop_val = 0.5; // 50% chance of "drop" (alpha scaling), 50% "keep" (beta scaling)
//
//     // Test ShakeDrop module directly
//     ShakeDrop sd_direct(p_drop_val);
//     std::cout << "ShakeDrop Module (direct use): " << sd_direct << std::endl;
//
//     torch::Tensor branch_res = torch::ones({1, 3, 4, 4}); // Example branch output
//     std::cout << std::fixed << std::setprecision(4);
//     std::cout << "Original branch output (sum): " << branch_res.sum().item<float>() << std::endl;
//
//     // --- Training mode ---
//     sd_direct->train();
//     std::cout << "\n--- Training Mode (Direct Use) ---" << std::endl;
//     for (int i = 0; i < 10; ++i) {
//         torch::Tensor output_train = sd_direct(branch_res);
//         std::cout << "Run " << i << ": Scaled branch sum = " << output_train.sum().item<float>() << std::endl;
//         // Expected: Sum will be original_sum * scale_factor.
//         // scale_factor is from U[-1,1] or U[0,2].
//     }
//
//     // --- Evaluation mode ---
//     sd_direct->eval();
//     torch::Tensor output_eval = sd_direct(branch_res);
//     std::cout << "\n--- Evaluation Mode (Direct Use) ---" << std::endl;
//     std::cout << "Scaled branch sum (eval) = " << output_eval.sum().item<float>() << std::endl;
//     // Expected sum: original_sum * (1 - p_drop_val) = (1*3*4*4) * (1 - 0.5) = 48 * 0.5 = 24.0
//     double expected_eval_sum = branch_res.sum().item<float>() * (1.0 - p_drop_val);
//     TORCH_CHECK(std::abs(output_eval.sum().item<float>() - expected_eval_sum) < 1e-4, "ShakeDrop eval output mismatch!");
//
//
//     // --- Test with ResidualBlockWithShakeDrop ---
//     std::cout << "\n--- ResidualBlockWithShakeDrop Test ---" << std::endl;
//     ResidualBlockWithShakeDrop res_block(3, p_drop_val);
//     std::cout << "ResidualBlock Module: " << res_block << std::endl;
//
//     torch::Tensor block_input = torch::randn({1, 3, 8, 8});
//     res_block->train();
//     std::cout << "Input sum: " << block_input.sum().item<float>() << std::endl;
//     std::cout << "Training mode (block will use stochastic ShakeDrop internally):" << std::endl;
//     for (int i=0; i < 3; ++i) {
//         torch::Tensor block_output_train = res_block(block_input);
//         std::cout << "  Run " << i << " Block output sum (train): " << block_output_train.sum().item<float>() << std::endl;
//     }
//
//     res_block->eval();
//     torch::Tensor block_output_eval = res_block(block_input);
//     std::cout << "Evaluation mode Block output sum: " << block_output_eval.sum().item<float>() << std::endl;
//     // The exact value here is harder to predict without knowing the conv weights,
//     // but the ShakeDrop component within it will be deterministic.
// }
//
// // int main() {
// //    run_shake_drop_example();
// //    return 0;
// // }
// */

namespace xt::dropouts
{
    ShakeDrop::ShakeDrop(double p_drop )
        : p_drop_(p_drop), gen_(std::random_device{}())
    {
        // Seed with random_device
        TORCH_CHECK(p_drop_ >= 0.0 && p_drop_ <= 1.0, "p_drop must be between 0 and 1.");
    }

    auto ShakeDrop::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        vector<std::any> tensors_ = tensors;
        auto branch_output = std::any_cast<torch::Tensor>(tensors_[0]);

        if (!this->is_training())
        {
            // Evaluation mode: scale by expected value of the forward scaling factor.
            // E[scaling_factor] = p_drop_ * E[alpha] + (1 - p_drop_) * E[beta]
            // E[alpha] = (alpha_range_min_ + alpha_range_max_) / 2.0 = (-1+1)/2 = 0
            // E[beta]  = (beta_range_min_ + beta_range_max_) / 2.0   = (0+2)/2 = 1
            // So, E[scaling_factor] = p_drop_ * 0 + (1 - p_drop_) * 1 = 1 - p_drop_
            return branch_output * (1.0 - p_drop_);
        }

        // Training mode: apply stochastic scaling
        double rand_val_for_choice = std::uniform_real_distribution<double>(0.0, 1.0)(gen_);
        double scale_factor;

        if (rand_val_for_choice < p_drop_)
        {
            // Apply alpha scaling (uniform from [-1, 1])
            scale_factor = std::uniform_real_distribution<double>(
                alpha_range_min_, alpha_range_max_
            )(gen_);
        }
        else
        {
            // Apply beta scaling (uniform from [0, 2])
            scale_factor = std::uniform_real_distribution<double>(
                beta_range_min_, beta_range_max_
            )(gen_);
        }

        // The scale_factor is a scalar, applied to the entire branch_output tensor.
        // The paper suggests per-channel or per-sample randomness for some variants (Shake-Shake),
        // but the core ShakeDrop applies a single random scalar per branch activation.
        return branch_output * scale_factor;
    }
}
