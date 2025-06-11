#include "include/normalizations/rezero.h"



// #include <torch/torch.h>
// #include <iostream>
// #include <vector>
//
// // Forward declaration for the Impl struct
// struct ReZeroImpl;
//
// // The main module struct that users will interact with.
// // This module represents the learnable 'alpha' scaling factor.
// struct ReZero : torch::nn::ModuleHolder<ReZeroImpl> {
//     using torch::nn::ModuleHolder<ReZeroImpl>::ModuleHolder;
//
//     // Takes F(x) (the output of the function in the residual block) as input
//     // and scales it by the learnable alpha.
//     torch::Tensor forward(torch::Tensor fx) {
//         return impl_->forward(fx);
//     }
// };
//
// // The implementation struct for ReZero's alpha scaling
// struct ReZeroImpl : torch::nn::Module {
//     // Learnable parameter 'alpha'
//     // Initialized to zero, as per the ReZero paper.
//     torch::Tensor alpha_;
//
//     ReZeroImpl(double initial_alpha_value = 0.0) {
//         // alpha_ is a learnable scalar parameter.
//         alpha_ = register_parameter("alpha", torch::tensor({initial_alpha_value}));
//     }
//
//     torch::Tensor forward(torch::Tensor fx) {
//         // fx: The output of the function F(x) within a residual block.
//         // This module scales fx by the learnable alpha.
//         // The residual connection x_skip + (alpha * fx) is typically handled outside this module.
//
//         return fx * alpha_;
//     }
//
//     void pretty_print(std::ostream& stream) const override {
//         stream << "ReZero(initial_alpha=" << alpha_.item<double>() << ")";
//         stream << "\n  (Note: This module provides alpha * F(x). The skip connection 'x + ...' is applied externally.)";
//     }
// };
// TORCH_MODULE(ReZero);
//
//
// // --- Example Usage ---
// // Example of a Residual Block using the ReZero module
// struct ResidualBlock : torch::nn::Module {
//     torch::nn::Linear fc1{nullptr}, fc2{nullptr};
//     ReZero rezero_scaler{nullptr}; // ReZero module for scaling F(x)
//
//     ResidualBlock(int64_t dim, double initial_rezero_alpha = 0.0) {
//         fc1 = register_module("fc1", torch::nn::Linear(dim, dim * 4));
//         fc2 = register_module("fc2", torch::nn::Linear(dim * 4, dim));
//         rezero_scaler = register_module("rezero_scaler", ReZero(initial_rezero_alpha));
//     }
//
//     torch::Tensor forward(torch::Tensor x_skip) {
//         // x_skip is the input to the residual block (the 'x' in x + alpha*F(x))
//
//         // F(x) part
//         torch::Tensor fx = fc1->forward(x_skip);
//         fx = torch::relu(fx);
//         fx = fc2->forward(fx);
//
//         // Apply ReZero scaling to F(x)
//         torch::Tensor scaled_fx = rezero_scaler->forward(fx);
//
//         // Add residual connection
//         return x_skip + scaled_fx;
//     }
// };
//
//
// int main() {
//     torch::manual_seed(0);
//
//     // --- Test Case 1: Basic ReZero module functionality ---
//     std::cout << "--- Test Case 1: ReZero module scaling ---" << std::endl;
//     double init_alpha1 = 0.0;
//     ReZero rz_module1(init_alpha1);
//     // std::cout << rz_module1 << std::endl;
//
//     torch::Tensor fx1 = torch::randn({4, 10}); // Example output F(x)
//     std::cout << "Input F(x) shape: " << fx1.sizes() << std::endl;
//     std::cout << "Initial alpha: " << rz_module1->alpha_.item<double>() << std::endl;
//
//     torch::Tensor scaled_fx1 = rz_module1->forward(fx1);
//     std::cout << "Output scaled_F(x) shape: " << scaled_fx1.sizes() << std::endl;
//     std::cout << "Output scaled_F(x) (should be all zeros initially): \n" << scaled_fx1 << std::endl;
//     TORCH_CHECK(torch::allclose(scaled_fx1, torch::zeros_like(fx1)),
//                 "Initial ReZero output not all zeros.");
//
//
//     // --- Test Case 2: Using ReZero within a ResidualBlock ---
//     std::cout << "\n--- Test Case 2: ReZero in a ResidualBlock ---" << std::endl;
//     int64_t dim = 10;
//     ResidualBlock res_block(dim, /*initial_rezero_alpha=*/0.0);
//     // std::cout << *res_block.named_modules()["rezero_scaler"] << std::endl;
//
//     torch::Tensor x_input_res = torch::randn({4, dim});
//     std::cout << "Input to ResidualBlock shape: " << x_input_res.sizes() << std::endl;
//
//     // Initially, alpha is 0, so output should be equal to input (x_skip + 0 * F(x_skip))
//     torch::Tensor res_output_initial = res_block.forward(x_input_res);
//     std::cout << "ResidualBlock output (initial alpha=0): \n" << res_output_initial << std::endl;
//     TORCH_CHECK(torch::allclose(res_output_initial, x_input_res),
//                 "ResidualBlock output with alpha=0 should be equal to input.");
//
//
//     // --- Test Case 3: Check backward pass and alpha update ---
//     std::cout << "\n--- Test Case 3: Backward pass and alpha update ---" << std::endl;
//     ResidualBlock res_block_train(dim, 0.0);
//     res_block_train.train(); // Ensure alpha has requires_grad=true
//
//     // Access the alpha parameter from the ReZero submodule
//     torch::Tensor& alpha_param = res_block_train.rezero_scaler->alpha_;
//     std::cout << "Initial alpha in res_block_train: " << alpha_param.item<double>() << std::endl;
//
//     // SGD optimizer for all parameters of res_block_train, including alpha
//     torch::optim::SGD optimizer(res_block_train.parameters(), /*lr=*/0.1);
//
//     optimizer.zero_grad();
//     torch::Tensor x_input_train = torch::randn({4, dim}, torch::requires_grad());
//     torch::Tensor res_output_train = res_block_train.forward(x_input_train);
//     torch::Tensor loss = res_output_train.mean(); // Simple loss
//     loss.backward();
//     optimizer.step();
//
//     std::cout << "Updated alpha in res_block_train: " << alpha_param.item<double>() << std::endl;
//     TORCH_CHECK(alpha_param.item<double>() != 0.0, "ReZero alpha did not update.");
//
//     bool grad_exists_x_input = x_input_train.grad().defined() &&
//                                x_input_train.grad().abs().sum().item<double>() > 0;
//     std::cout << "Gradient exists for x_input_train: " << (grad_exists_x_input ? "true" : "false") << std::endl;
//     TORCH_CHECK(grad_exists_x_input, "No gradient for block input x_input_train!");
//     TORCH_CHECK(alpha_param.grad().defined(), "Alpha should have a gradient after backward.");
//
//
//     // Test with non-zero initial alpha
//     std::cout << "\n--- Test Case 4: Non-zero initial alpha ---" << std::endl;
//     ReZero rz_module4(0.5);
//     torch::Tensor fx4 = torch::tensor({1.0, 2.0, 3.0});
//     torch::Tensor scaled_fx4 = rz_module4->forward(fx4);
//     std::cout << "Input F(x): " << fx4 << std::endl;
//     std::cout << "Alpha: " << rz_module4->alpha_.item<double>() << std::endl;
//     std::cout << "Output scaled_F(x): " << scaled_fx4 << std::endl;
//     TORCH_CHECK(torch::allclose(scaled_fx4, fx4 * 0.5), "Scaling with non-zero alpha failed.");
//
//
//     std::cout << "\nReZero tests finished." << std::endl;
//     return 0;
// }



namespace xt::norm
{
    auto Rezero::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
