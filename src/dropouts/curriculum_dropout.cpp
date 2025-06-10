#include "include/dropouts/curriculum_dropout.h"

//
// #include <torch/torch.h>
// #include <vector>
// #include <algorithm> // For std::min, std::max
// #include <ostream>   // For std::ostream
//
// struct CurriculumDropoutImpl : torch::nn::Module {
//     double initial_dropout_rate_;
//     double final_dropout_rate_;
//     int64_t total_curriculum_steps_; // Number of steps/epochs over which the rate changes
//     bool increase_rate_; // If true, rate goes from initial to final. If false, from final to initial.
//
//     // A transient member to store the most recently calculated dropout rate for pretty_print or inspection
//     // Not a registered parameter or buffer.
//     mutable double current_p_ = 0.0;
//
//
//     CurriculumDropoutImpl(
//         double initial_dropout_rate = 0.5,
//         double final_dropout_rate = 0.1,
//         int64_t total_curriculum_steps = 10000, // e.g., total training steps or epochs for curriculum
//         bool increase_rate = false // Default: decrease rate from initial to final
//     ) : initial_dropout_rate_(initial_dropout_rate),
//         final_dropout_rate_(final_dropout_rate),
//         total_curriculum_steps_(std::max(int64_t(1), total_curriculum_steps)), // Ensure at least 1 step
//         increase_rate_(increase_rate)
//     {
//         TORCH_CHECK(initial_dropout_rate_ >= 0.0 && initial_dropout_rate_ <= 1.0, "Initial dropout rate must be between 0 and 1.");
//         TORCH_CHECK(final_dropout_rate_ >= 0.0 && final_dropout_rate_ <= 1.0, "Final dropout rate must be between 0 and 1.");
//     }
//
//     // The forward method needs to know the current progress in the curriculum.
//     // current_step could be the global training step or current epoch, depending on how
//     // total_curriculum_steps_ is defined.
//     torch::Tensor forward(const torch::Tensor& input, int64_t current_step) {
//         if (!this->is_training()) {
//             current_p_ = 0.0; // No dropout during evaluation
//             return input;
//         }
//
//         double start_rate, end_rate;
//         if (increase_rate_) {
//             start_rate = initial_dropout_rate_;
//             end_rate = final_dropout_rate_;
//         } else {
//             start_rate = initial_dropout_rate_; // Higher rate at the start
//             end_rate = final_dropout_rate_;     // Lower rate at the end
//         }
//
//         // Calculate progress ratio (0.0 to 1.0)
//         double progress_ratio = static_cast<double>(current_step) / static_cast<double>(total_curriculum_steps_);
//         progress_ratio = std::min(1.0, std::max(0.0, progress_ratio)); // Clamp to [0, 1]
//
//         // Linear interpolation for the dropout rate
//         current_p_ = start_rate + (end_rate - start_rate) * progress_ratio;
//
//         // Ensure current_p_ is valid (it should be if start/end rates are valid)
//         current_p_ = std::min(1.0, std::max(0.0, current_p_));
//
//         if (current_p_ == 0.0) {
//             return input;
//         }
//         if (current_p_ == 1.0) {
//             return torch::zeros_like(input);
//         }
//
//         // Standard inverted dropout
//         double keep_prob = 1.0 - current_p_;
//         torch::Tensor mask = (torch::rand_like(input) < keep_prob).to(input.dtype());
//
//         return (input * mask) / keep_prob;
//     }
//
//     double get_current_dropout_rate() const {
//         return current_p_;
//     }
//
//     void pretty_print(std::ostream& stream) const override {
//         stream << "CurriculumDropout(initial_p=" << initial_dropout_rate_
//                << ", final_p=" << final_dropout_rate_
//                << ", total_steps=" << total_curriculum_steps_
//                << ", increase_rate=" << (increase_rate_ ? "true" : "false")
//                << ", last_calculated_p=" << current_p_ // Show p from last forward if available
//                << ")";
//     }
// };
//
// TORCH_MODULE(CurriculumDropout); // Creates the CurriculumDropout module "class"
//
// /*
// // Example of how to use the CurriculumDropout module:
// // (This is for illustration and would typically be in your main application code)
//
// #include <iostream>
//
// void run_curriculum_dropout_example() {
//     torch::manual_seed(0); // For reproducible results
//
//     double initial_p = 0.5;
//     double final_p = 0.05; // Target a lower dropout rate eventually
//     int64_t curriculum_duration_steps = 100;
//
//     CurriculumDropout dropout_module(initial_p, final_p, curriculum_duration_steps, false); // Decrease rate
//     std::cout << "Curriculum Dropout Module: " << dropout_module << std::endl;
//
//     torch::Tensor input_tensor = torch::ones({2, 5}); // Example input
//
//     dropout_module->train(); // Set to training mode
//
//     std::cout << "\n--- Simulating Training Steps ---" << std::endl;
//     for (int64_t step : {0, curriculum_duration_steps / 2, curriculum_duration_steps, curriculum_duration_steps + 50}) {
//         torch::Tensor output = dropout_module->forward(input_tensor, step);
//         std::cout << "Step: " << step
//                   << ", Current Dropout P: " << dropout_module->get_current_dropout_rate()
//                   << std::endl;
//         // std::cout << "Output at step " << step << ":\n" << output << std::endl; // Optional: print output
//         if (step == curriculum_duration_steps / 2) {
//             // Rate should be around (0.5 + 0.05) / 2 = 0.275
//             TORCH_CHECK(std::abs(dropout_module->get_current_dropout_rate() - (initial_p + final_p) / 2.0) < 1e-6, "Rate mismatch at mid-point");
//         } else if (step >= curriculum_duration_steps) {
//             // Rate should be final_p
//             TORCH_CHECK(std::abs(dropout_module->get_current_dropout_rate() - final_p) < 1e-6, "Rate mismatch at/after end-point");
//         }
//     }
//
//     // --- Evaluation mode ---
//     dropout_module->eval(); // Set to evaluation mode
//     torch::Tensor output_eval = dropout_module->forward(input_tensor, curriculum_duration_steps + 100); // Step value doesn't matter in eval
//     std::cout << "\nOutput (evaluation mode):\n" << output_eval << std::endl;
//     // Expected output to be identical to input in evaluation mode.
//     TORCH_CHECK(torch::allclose(output_eval, input_tensor), "Output in eval mode should be input.");
//     std::cout << "Current Dropout P (eval mode): " << dropout_module->get_current_dropout_rate() << std::endl; // Should be 0.0
//
//     // Example: Increasing rate
//     CurriculumDropout increasing_dropout_module(0.05, 0.5, curriculum_duration_steps, true); // Increase rate
//     std::cout << "\nIncreasing Rate Curriculum Dropout Module: " << increasing_dropout_module << std::endl;
//     increasing_dropout_module->train();
//     increasing_dropout_module->forward(input_tensor, 0);
//     std::cout << "Step: 0, Current Dropout P: " << increasing_dropout_module->get_current_dropout_rate() << std::endl;
//     TORCH_CHECK(std::abs(increasing_dropout_module->get_current_dropout_rate() - 0.05) < 1e-6, "Rate mismatch for increasing");
//     increasing_dropout_module->forward(input_tensor, curriculum_duration_steps);
//     std::cout << "Step: " << curriculum_duration_steps << ", Current Dropout P: " << increasing_dropout_module->get_current_dropout_rate() << std::endl;
//     TORCH_CHECK(std::abs(increasing_dropout_module->get_current_dropout_rate() - 0.5) < 1e-6, "Rate mismatch for increasing");
//
//
// }
//
// // int main() {
// //    run_curriculum_dropout_example();
// //    return 0;
// // }
// */

namespace xt::dropouts
{
    torch::Tensor curriculum_dropout(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto CurriculumDropout::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::dropouts::curriculum_dropout(torch::zeros(10));
    }
}
