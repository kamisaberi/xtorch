#include "include/dropouts/scheduled_drop_path.h"



#include <torch/torch.h>
#include <vector>
#include <algorithm> // For std::min, std::max
#include <ostream>   // For std::ostream

struct ScheduledDropPathImpl : torch::nn::Module {
    double initial_drop_rate_;
    double final_drop_rate_;
    int64_t total_curriculum_steps_; // Number of steps/epochs over which the rate changes
    bool increase_rate_;             // If true, rate goes from initial to final.
                                     // Common for DropPath: start low (or 0), increase to final.
    double epsilon_ = 1e-7;          // For numerical stability

    // Transient member to store the most recently calculated drop rate
    mutable double current_p_drop_ = 0.0;

    ScheduledDropPathImpl(
        double final_drop_rate = 0.2,            // The target drop rate at the end of the schedule
        int64_t total_curriculum_steps = 100000, // Total steps for the schedule
        double initial_drop_rate = 0.0           // Starting drop rate (often 0 for DropPath)
    ) : initial_drop_rate_(initial_drop_rate),
        final_drop_rate_(final_drop_rate),
        total_curriculum_steps_(std::max(int64_t(1), total_curriculum_steps)),
        increase_rate_(final_drop_rate_ > initial_drop_rate_) // Automatically determine if increasing
    {
        TORCH_CHECK(initial_drop_rate_ >= 0.0 && initial_drop_rate_ <= 1.0, "Initial drop rate must be between 0 and 1.");
        TORCH_CHECK(final_drop_rate_ >= 0.0 && final_drop_rate_ <= 1.0, "Final drop rate must be between 0 and 1.");
    }

    // The forward method needs the current training step to calculate the current drop rate.
    torch::Tensor forward(const torch::Tensor& input, int64_t current_step) {
        if (!this->is_training()) {
            current_p_drop_ = 0.0; // No dropout during evaluation
            return input;
        }

        double start_rate = initial_drop_rate_;
        double end_rate = final_drop_rate_;

        // Calculate progress ratio (0.0 to 1.0)
        double progress_ratio = static_cast<double>(current_step) / static_cast<double>(total_curriculum_steps_);
        progress_ratio = std::min(1.0, std::max(0.0, progress_ratio)); // Clamp to [0, 1]

        // Linear interpolation for the drop rate
        current_p_drop_ = start_rate + (end_rate - start_rate) * progress_ratio;
        current_p_drop_ = std::min(1.0, std::max(0.0, current_p_drop_)); // Ensure valid probability

        if (current_p_drop_ == 0.0) {
            return input;
        }
        if (current_p_drop_ == 1.0) {
            return torch::zeros_like(input);
        }

        TORCH_CHECK(input.dim() >= 1, "ScheduledDropPath input must have at least one dimension (expected batch dimension at dim 0).");

        int64_t batch_size = input.size(0);
        double keep_prob = 1.0 - current_p_drop_;

        // Create a per-sample mask (1 to keep, 0 to drop)
        torch::Tensor random_tensor = torch::rand({batch_size}, input.options());
        torch::Tensor keep_mask_1d = (random_tensor < keep_prob).to(input.dtype());

        // Reshape mask to be broadcastable with input: (N, 1, 1, ...)
        std::vector<int64_t> view_shape(input.dim(), 1L);
        if (input.dim() > 0) {
            view_shape[0] = batch_size;
        } else { // Should not happen
             return input;
        }
        torch::Tensor keep_mask = keep_mask_1d.view(view_shape);

        return (input * keep_mask) / (keep_prob + epsilon_);
    }

    double get_current_drop_rate() const {
        return current_p_drop_;
    }

    void pretty_print(std::ostream& stream) const override {
        stream << "ScheduledDropPath(initial_p=" << initial_drop_rate_
               << ", final_p=" << final_drop_rate_
               << ", total_steps=" << total_curriculum_steps_
               << ", increase_rate_implied=" << (increase_rate_ ? "true" : "false")
               << ", last_calculated_p_drop=" << current_p_drop_
               << ")";
    }
};

TORCH_MODULE(ScheduledDropPath); // Creates the ScheduledDropPath module "class"

/*
// Example of how to use the ScheduledDropPath module:
// (This is for illustration and would typically be in your main application code)

#include <iostream>

// --- A simple example of a "pathway" (e.g., a residual block's output) ---
struct ExampleBlockWithScheduledDropPath : torch::nn::Module {
    torch::nn::Linear fc{nullptr};
    ScheduledDropPath scheduled_drop_path_module;

    ExampleBlockWithScheduledDropPath(int features, double final_dp_rate, int64_t total_steps)
        : scheduled_drop_path_module(final_dp_rate, total_steps, 0.0) // Start drop rate at 0
    {
        fc = register_module("fc", torch::nn::Linear(features, features));
        // scheduled_drop_path_module is already initialized and will be registered if named
        // For direct use as a member, manual registration is not strictly needed unless
        // you want it to appear in model.modules() or model.parameters() (if it had params).
        // Since it has no learnable parameters, it's fine as a direct member.
    }

    // The block's forward method now needs current_step
    torch::Tensor forward(const torch::Tensor& x, int64_t current_step) {
        torch::Tensor identity = x;
        torch::Tensor out = torch::relu(fc(x)); // Some computation

        // Apply ScheduledDropPath to the output of this "pathway"
        out = scheduled_drop_path_module(out, current_step); // Pass current_step

        out += identity; // Add skip connection
        return torch::relu(out);
    }
};
TORCH_MODULE(ExampleBlockWithScheduledDropPath);


void run_scheduled_drop_path_example() {
    torch::manual_seed(1); // For reproducible results

    double target_final_drop_rate = 0.3;
    int64_t curriculum_total_steps = 100; // Schedule completes over 100 steps
    double initial_drop_rate = 0.0; // Start with no drop path

    // Test ScheduledDropPath directly
    ScheduledDropPath drop_path_direct(target_final_drop_rate, curriculum_total_steps, initial_drop_rate);
    std::cout << "ScheduledDropPath Module (direct use): " << drop_path_direct << std::endl;

    torch::Tensor input_tensor = torch::ones({4, 2, 2}); // Batch=4, Seq=2, Feat=2
    input_tensor[1] *= 2.0;
    input_tensor[2] *= 3.0;
    input_tensor[3] *= 4.0;

    drop_path_direct->train(); // Set to training mode

    std::cout << "\n--- Simulating Training Steps (Direct Use) ---" << std::endl;
    for (int64_t step : {0LL, curriculum_total_steps / 2, curriculum_total_steps, curriculum_total_steps + 20}) {
        torch::Tensor output = drop_path_direct(input_tensor, step);
        std::cout << "Step: " << step
                  << ", Current DropPath P: " << drop_path_direct.get_current_drop_rate()
                  << std::endl;
        // Check sums to see effect (optional)
        // for (int i = 0; i < output.size(0); ++i) {
        //     std::cout << "  Output sample " << i << " sum: " << output[i].sum().item<float>() << std::endl;
        // }

        if (step == 0) {
            TORCH_CHECK(std::abs(drop_path_direct.get_current_drop_rate() - initial_drop_rate) < 1e-6, "Rate mismatch at step 0");
        } else if (step == curriculum_total_steps / 2) {
            double expected_mid_rate = initial_drop_rate + (target_final_drop_rate - initial_drop_rate) * 0.5;
            TORCH_CHECK(std::abs(drop_path_direct.get_current_drop_rate() - expected_mid_rate) < 1e-6, "Rate mismatch at mid-point");
        } else if (step >= curriculum_total_steps) {
            TORCH_CHECK(std::abs(drop_path_direct.get_current_drop_rate() - target_final_drop_rate) < 1e-6, "Rate mismatch at/after end-point");
        }
    }

    drop_path_direct->eval();
    torch::Tensor output_direct_eval = drop_path_direct(input_tensor, curriculum_total_steps + 50);
    TORCH_CHECK(torch::allclose(input_tensor, output_direct_eval), "ScheduledDropPath direct eval mismatch");
    std::cout << "Current DropPath P (eval mode): " << drop_path_direct.get_current_drop_rate() << std::endl;


    // --- Test with ExampleBlockWithScheduledDropPath ---
    std::cout << "\n--- ExampleBlockWithScheduledDropPath Test ---" << std::endl;
    int features = 2 * 2; // input_tensor.size(1) * input_tensor.size(2)
    ExampleBlockWithScheduledDropPath block_module(features, target_final_drop_rate, curriculum_total_steps);
    std::cout << "ExampleBlock Module: " << block_module << std::endl;

    block_module->train();
    torch::Tensor block_input = input_tensor.view({input_tensor.size(0), -1}); // Flatten for Linear layer

    for (int64_t step : {0LL, curriculum_total_steps / 2, curriculum_total_steps}) {
        torch::Tensor block_output = block_module(block_input, step);
        std::cout << "Step: " << step
                  << ", Block's DropPath P: " << block_module->scheduled_drop_path_module.get_current_drop_rate()
                  << ", Output sum (sample 0): " << block_output[0].sum().item<float>()
                  << std::endl;
    }
}

// int main() {
//    run_scheduled_drop_path_example();
//    return 0;
// }
*/


namespace xt::dropouts
{
    torch::Tensor scheduled_drop_path(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto ScheduledDropPath::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::dropouts::scheduled_drop_path(torch::zeros(10));
    }
}
