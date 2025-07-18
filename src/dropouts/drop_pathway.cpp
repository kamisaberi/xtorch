#include <dropouts/drop_pathway.h>

// #include <torch/torch.h>
// #include <vector>
// #include <ostream> // For std::ostream
//
// struct DropPathwayImpl : torch::nn::Module {
//     double p_drop_; // Probability of dropping the pathway (input tensor)
//     double epsilon_ = 1e-7; // For numerical stability in division
//
//     DropPathwayImpl(double p_drop = 0.1) : p_drop_(p_drop) {
//         TORCH_CHECK(p_drop_ >= 0.0 && p_drop_ <= 1.0, "DropPathway probability p_drop must be between 0 and 1.");
//     }
//
//     torch::Tensor forward(const torch::Tensor& input) {
//         if (!this->is_training() || p_drop_ == 0.0) {
//             return input; // Pass through if not training or no dropout
//         }
//         if (p_drop_ == 1.0) {
//             return torch::zeros_like(input); // Drop everything if p_drop is 1.0
//         }
//
//         TORCH_CHECK(input.dim() >= 1, "DropPathway input must have at least one dimension (expected batch dimension at dim 0).");
//
//         int64_t batch_size = input.size(0);
//         double keep_prob = 1.0 - p_drop_;
//
//         // Create a per-sample binary mask (1 for keep, 0 for drop)
//         // This mask is applied to each sample in the batch independently.
//         torch::Tensor random_tensor = torch::rand({batch_size}, input.options());
//         torch::Tensor keep_mask_1d = (random_tensor < keep_prob).to(input.dtype());
//
//         // Reshape mask to be broadcastable with the input tensor.
//         // If input is (N, C, H, W), mask becomes (N, 1, 1, 1).
//         std::vector<int64_t> view_shape(input.dim(), 1L);
//         if (input.dim() > 0) { // Should always be true due to TORCH_CHECK above
//             view_shape[0] = batch_size;
//         } else { // Defensive coding, should not be reached
//              return input;
//         }
//         torch::Tensor keep_mask = keep_mask_1d.view(view_shape);
//
//         // Apply the mask and scale the output (inverted dropout).
//         // Kept pathways are scaled by 1/keep_prob.
//         return (input * keep_mask) / (keep_prob + epsilon_);
//     }
//
//     void pretty_print(std::ostream& stream) const override {
//         stream << "DropPathway(p_drop=" << p_drop_ << ")";
//     }
// };
//
// TORCH_MODULE(DropPathway); // Creates the DropPathway module "class"
//
// /*
// // Example of how to use the DropPathway module:
// // (This is for illustration and would typically be in your main application code)
//
// #include <iostream>
//
// // --- A simple example of a "pathway" (e.g., a residual block) ---
// struct ExamplePathway : torch::nn::Module {
//     torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
//     DropPathway drop_pathway_module; // Instance of DropPathway
//
//     ExamplePathway(int channels, double dp_rate) : drop_pathway_module(dp_rate) {
//         conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3).padding(1)));
//         conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3).padding(1)));
//         // drop_pathway_module is already initialized
//     }
//
//     torch::Tensor forward(const torch::Tensor& x) {
//         torch::Tensor identity = x;
//         torch::Tensor out = torch::relu(conv1(x));
//         out = conv2(out);
//
//         // Apply DropPathway to the output of the "pathway" (conv operations)
//         // before adding the identity.
//         if (this->is_training()) { // DropPathway is only active in training
//             out = drop_pathway_module(out);
//         }
//         // During eval, drop_pathway_module(out) will just return 'out'.
//
//         out += identity; // Add skip connection
//         return torch::relu(out);
//     }
// };
// TORCH_MODULE(ExamplePathway);
//
//
// void run_drop_pathway_example() {
//     torch::manual_seed(42); // For reproducible results
//
//     double drop_probability = 0.5;
//
//     // Test DropPathway directly
//     DropPathway direct_dp_module(drop_probability);
//     std::cout << "Direct DropPathway Module: " << direct_dp_module << std::endl;
//
//     torch::Tensor input_tensor = torch::ones({4, 2, 3, 3}); // Batch=4, C=2, H=3, W=3
//     input_tensor[0] *= 1.0;
//     input_tensor[1] *= 2.0;
//     input_tensor[2] *= 3.0;
//     input_tensor[3] *= 4.0;
//
//     std::cout << "\nInput Tensor (shape " << input_tensor.sizes() << "):" << std::endl;
//     for (int i = 0; i < input_tensor.size(0); ++i) {
//         std::cout << "Input sample " << i << " sum: " << input_tensor[i].sum().item<float>() << std::endl;
//     }
//
//     direct_dp_module->train();
//     torch::Tensor output_direct_train = direct_dp_module->forward(input_tensor);
//     std::cout << "\nOutput from direct DropPathway (training, p_drop=" << drop_probability << "):" << std::endl;
//     for (int i = 0; i < output_direct_train.size(0); ++i) {
//         std::cout << "Output sample " << i << " sum: " << output_direct_train[i].sum().item<float>()
//                   << (output_direct_train[i].sum().item<float>() == 0 ? " (Dropped)" : " (Kept & Scaled)")
//                   << std::endl;
//     }
//     // Expected: Approx 50% of samples (entire BxCxHxW slice) will be zeroed out.
//     // Non-zero samples will be scaled by 1 / (1 - 0.5) = 2.
//
//     direct_dp_module->eval();
//     torch::Tensor output_direct_eval = direct_dp_module->forward(input_tensor);
//     std::cout << "\nOutput from direct DropPathway (evaluation):" << std::endl;
//     for (int i = 0; i < output_direct_eval.size(0); ++i) {
//         std::cout << "Output sample " << i << " sum: " << output_direct_eval[i].sum().item<float>() << std::endl;
//     }
//     TORCH_CHECK(torch::allclose(input_tensor, output_direct_eval), "Direct DropPathway eval output mismatch!");
//
//
//     // --- Test with the ExamplePathway that uses DropPathway internally ---
//     std::cout << "\n--- Testing ExamplePathway with internal DropPathway ---" << std::endl;
//     int channels = 2;
//     ExamplePathway pathway_block(channels, drop_probability);
//     std::cout << "ExamplePathway block (contains DropPathway): " << pathway_block << std::endl;
//
//     pathway_block->train(); // Set the outer block (and thus inner DropPathway) to training mode
//     torch::Tensor block_output_train = pathway_block->forward(input_tensor);
//     std::cout << "Output from ExamplePathway (training mode):" << std::endl;
//     // The effect of DropPathway here will be on the 'out' tensor within ExamplePathway's forward,
//     // before the residual connection is added. Some samples' 'out' will be zeroed.
//     for (int i = 0; i < block_output_train.size(0); ++i) {
//         // If pathway was dropped for sample i, block_output_train[i] should be relu(identity[i])
//         // If pathway was kept, block_output_train[i] should be relu(identity[i] + scaled_conv_out[i])
//         std::cout << "ExamplePathway Output sample " << i << " sum: " << block_output_train[i].sum().item<float>() << std::endl;
//     }
//
//
//     pathway_block->eval(); // Set to evaluation mode
//     torch::Tensor block_output_eval = pathway_block->forward(input_tensor);
//     std::cout << "\nOutput from ExamplePathway (evaluation mode):" << std::endl;
//      for (int i = 0; i < block_output_eval.size(0); ++i) {
//         std::cout << "ExamplePathway Output sample " << i << " sum: " << block_output_eval[i].sum().item<float>() << std::endl;
//     }
//     // In eval mode, DropPathway is identity, so it should be a standard residual block operation.
// }
//
// // int main() {
// //    run_drop_pathway_example();
// //    return 0;
// // }
// */

namespace xt::dropouts
{
    DropPathway::DropPathway(double p_drop) : p_drop_(p_drop)
    {
        TORCH_CHECK(p_drop_ >= 0.0 && p_drop_ <= 1.0, "DropPathway probability p_drop must be between 0 and 1.");
    }

    auto DropPathway::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        vector<std::any> tensors_ = tensors;
        auto input = std::any_cast<torch::Tensor>(tensors_[0]);


        if (!this->is_training() || p_drop_ == 0.0)
        {
            return input; // Pass through if not training or no dropout
        }
        if (p_drop_ == 1.0)
        {
            return torch::zeros_like(input); // Drop everything if p_drop is 1.0
        }

        TORCH_CHECK(input.dim() >= 1,
                    "DropPathway input must have at least one dimension (expected batch dimension at dim 0).");

        int64_t batch_size = input.size(0);
        double keep_prob = 1.0 - p_drop_;

        // Create a per-sample binary mask (1 for keep, 0 for drop)
        // This mask is applied to each sample in the batch independently.
        torch::Tensor random_tensor = torch::rand({batch_size}, input.options());
        torch::Tensor keep_mask_1d = (random_tensor < keep_prob).to(input.dtype());

        // Reshape mask to be broadcastable with the input tensor.
        // If input is (N, C, H, W), mask becomes (N, 1, 1, 1).
        std::vector<int64_t> view_shape(input.dim(), 1L);
        if (input.dim() > 0)
        {
            // Should always be true due to TORCH_CHECK above
            view_shape[0] = batch_size;
        }
        else
        {
            // Defensive coding, should not be reached
            return input;
        }
        torch::Tensor keep_mask = keep_mask_1d.view(view_shape);

        // Apply the mask and scale the output (inverted dropout).
        // Kept pathways are scaled by 1/keep_prob.
        return (input * keep_mask) / (keep_prob + epsilon_);
    }
}
