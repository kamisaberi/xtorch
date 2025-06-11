#include "include/normalizations/self_norm.h"



// #include <torch/torch.h>
// #include <iostream>
// #include <vector>
// #include <cmath> // For std::exp, std::max, std::min (though torch functions are better)
//
// // Forward declaration for the Impl struct
// struct SelfNormActivationImpl;
//
// // The main module struct that users will interact with.
// // This module applies the SELU activation, which is central to Self-Normalizing Networks.
// // It doesn't "normalize" in the traditional sense of BatchNorm, but enables self-normalization.
// struct SelfNormActivation : torch::nn::ModuleHolder<SelfNormActivationImpl> {
//     using torch::nn::ModuleHolder<SelfNormActivationImpl>::ModuleHolder;
//
//     torch::Tensor forward(torch::Tensor x) {
//         return impl_->forward(x);
//     }
// };
//
// // The implementation struct for SelfNormActivation (applying SELU)
// struct SelfNormActivationImpl : torch::nn::Module {
//     // SELU has fixed alpha and scale parameters.
//     // These are constants derived for self-normalizing properties.
//     // alpha = 1.6732632423543772848170429916717
//     // scale = 1.0507009873554804934193349852946
//     // LibTorch's torch::selu function uses these constants internally.
//
//     bool inplace_; // Whether to perform the operation in-place
//
//     SelfNormActivationImpl(bool inplace = false) : inplace_(inplace) {
//         // No learnable parameters for SELU itself.
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // Apply the SELU activation function.
//         // torch::selu(input, inplace)
//         // If using nn::SELUM(), it also has an inplace option.
//
//         if (inplace_) {
//             return torch::selu_(x); // In-place version
//         } else {
//             return torch::selu(x);  // Out-of-place version
//         }
//     }
//
//     void pretty_print(std::ostream& stream) const override {
//         stream << "SelfNormActivation(SELU, inplace=" << (inplace_ ? "true" : "false") << ")";
//         stream << "\n  (Note: Enables self-normalizing properties when used with appropriate weight initialization.)";
//     }
// };
// TORCH_MODULE(SelfNormActivation);
//
//
// // --- Example Usage ---
// int main() {
//     torch::manual_seed(0);
//
//     // --- Test Case 1: SelfNormActivation (SELU) basic functionality ---
//     std::cout << "--- Test Case 1: SelfNormActivation (SELU) ---" << std::endl;
//     SelfNormActivation selu_module_out_of_place; // Default: out-of-place
//     SelfNormActivation selu_module_in_place(true);   // In-place
//     // std::cout << selu_module_out_of_place << std::endl;
//
//     torch::Tensor x1 = torch::tensor({-2.0, -0.5, 0.0, 0.5, 2.0});
//     std::cout << "Input x1: " << x1 << std::endl;
//
//     // SELU constants
//     const double alpha = 1.6732632423543772848170429916717;
//     const double scale = 1.0507009873554804934193349852946;
//
//     // Manual calculation for x = -2.0:
//     // scale * (alpha * (exp(-2.0) - 1.0))
//     // = 1.0507... * (1.6732... * (0.1353... - 1.0))
//     // = 1.0507... * (1.6732... * -0.8646...)
//     // = 1.0507... * -1.446...
//     // = -1.519...
//     // Manual calculation for x = 2.0:
//     // scale * 2.0 = 1.0507... * 2.0 = 2.1014...
//
//     torch::Tensor y1_out = selu_module_out_of_place->forward(x1);
//     std::cout << "Output y1 (out-of-place SELU): " << y1_out << std::endl;
//
//     torch::Tensor x1_copy_for_inplace = x1.clone();
//     torch::Tensor y1_in = selu_module_in_place->forward(x1_copy_for_inplace);
//     std::cout << "Output y1 (in-place SELU, from modified x1_copy): " << y1_in << std::endl;
//     std::cout << "x1_copy_for_inplace after in-place op: " << x1_copy_for_inplace << std::endl;
//
//     TORCH_CHECK(torch::allclose(y1_out, y1_in), "In-place and out-of-place SELU results differ.");
//     TORCH_CHECK(std::abs(y1_out[0].item<double>() - (-1.5199295)) < 1e-5, "SELU output for -2.0 mismatch.");
//     TORCH_CHECK(std::abs(y1_out[4].item<double>() - (2.1014019)) < 1e-5, "SELU output for 2.0 mismatch.");
//
//
//     // --- Test Case 2: Using in a simple network context (conceptual) ---
//     // For SNNs to work, weight initialization (e.g., LeCun normal for weights of dense layers)
//     // is also critical. This module only provides the activation.
//     std::cout << "\n--- Test Case 2: Conceptual SNN layer ---" << std::endl;
//     struct SimpleSnnLayer : torch::nn::Module {
//         torch::nn::Linear linear{nullptr};
//         SelfNormActivation selu_act{nullptr};
//
//         SimpleSnnLayer(int64_t in, int64_t out) {
//             linear = register_module("linear", torch::nn::Linear(in, out));
//             selu_act = register_module("selu_act", SelfNormActivation());
//
//             // For SNNs, specific weight initialization is important for `linear` layer.
//             // E.g., for dense layers, weights from N(0, 1/fan_in) and biases to 0.
//             // torch::nn::init::normal_(linear->weight, 0, std::sqrt(1.0 / static_cast<double>(in)));
//             // if (linear->bias.defined()) torch::nn::init::zeros_(linear->bias);
//             // This part is illustrative of what else is needed for SNNs.
//         }
//
//         torch::Tensor forward(torch::Tensor x) {
//             x = linear->forward(x);
//             x = selu_act->forward(x);
//             return x;
//         }
//     };
//
//     SimpleSnnLayer snn_layer(10, 20);
//     torch::Tensor x2 = torch::randn({4, 10}); // Batch of 4, 10 features
//     std::cout << "Input x2 shape: " << x2.sizes() << std::endl;
//     torch::Tensor y2 = snn_layer.forward(x2);
//     std::cout << "Output y2 shape: " << y2.sizes() << std::endl;
//     std::cout << "Output y2 mean: " << y2.mean().item<double>() << std::endl;
//     std::cout << "Output y2 std: " << y2.std().item<double>() << std::endl;
//     // For true self-normalization, after many layers and with correct init, mean should be ~0 and std ~1.
//     // This single layer example won't fully demonstrate that property without proper init and depth.
//
//
//     // --- Test Case 3: Check backward pass ---
//     // SELU is differentiable.
//     std::cout << "\n--- Test Case 3: Backward pass check ---" << std::endl;
//     SelfNormActivation selu_module3;
//     selu_module3.train(); // Mode doesn't change SELU behavior, but good practice
//
//     torch::Tensor x3 = torch::randn({4, 5}, torch::requires_grad());
//     torch::Tensor y3 = selu_module3->forward(x3);
//     torch::Tensor loss = y3.mean();
//     loss.backward();
//
//     bool grad_exists_x3 = x3.grad().defined() && x3.grad().abs().sum().item<double>() > 0;
//     std::cout << "Gradient exists for x3: " << (grad_exists_x3 ? "true" : "false") << std::endl;
//     TORCH_CHECK(grad_exists_x3, "No gradient for x3!");
//
//     auto params = selu_module3->parameters();
//     std::cout << "Number of learnable parameters: " << params.size() << std::endl;
//     TORCH_CHECK(params.empty(), "SelfNormActivation (SELU) should have no learnable parameters.");
//
//
//     std::cout << "\nSelfNormActivation (SELU) tests finished." << std::endl;
//     return 0;
// }



namespace xt::norm
{
    auto SelfNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
