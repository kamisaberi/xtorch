#include <dropouts/attention_dropout.h>


// #include <torch/torch.h>
// #include <ostream> // For std::ostream
//
// // No specific paper reference is typically needed for standard dropout,
// // as it's a foundational technique. If applying it in a novel way within
// // attention, the paper introducing that specific attention mechanism would be relevant.
// // Original Dropout Paper:
// // Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014).
// // Dropout: A simple way to prevent neural networks from overfitting.
// // Journal of Machine Learning Research, 15(1), 1929-1958.
// // Link: http://jmlr.org/papers/v15/srivastava14a.html
//
// struct AttentionDropoutImpl : torch::nn::Module {
//     double p_; // Probability of an element to be zeroed.
//
//     AttentionDropoutImpl(double p = 0.1) : p_(p) {
//         TORCH_CHECK(p_ >= 0.0 && p_ <= 1.0, "Dropout probability has to be between 0 and 1, but got ", p_);
//     }
//
//     torch::Tensor forward(const torch::Tensor& input) {
//         if (!this->is_training() || p_ == 0.0) {
//             return input;
//         }
//
//         // If p_ is 1.0, all elements are dropped.
//         if (p_ == 1.0) {
//             return torch::zeros_like(input);
//         }
//
//         // Create a mask where elements are 1 with probability (1.0 - p_)
//         // and 0 with probability p_.
//         // Then, scale the result by 1.0 / (1.0 - p_).
//         // This is the "inverted dropout" technique.
//         torch::Tensor mask = (torch::rand_like(input) < (1.0 - p_)).to(input.dtype());
//
//         // Ensure denominator is not zero (already handled by p_ == 1.0 check, but good for clarity)
//         return (input * mask) / (1.0 - p_);
//     }
//
//     void pretty_print(std::ostream& stream) const override {
//         stream << "AttentionDropout(p=" << p_ << ")";
//     }
// };
//
// TORCH_MODULE(AttentionDropout); // Creates the AttentionDropout module "class"
//
// /*
// // Example of how to use the AttentionDropout module:
// // (This is for illustration and would typically be in your main application code)
//
// #include <iostream>
//
// void run_attention_dropout_example() {
//     torch::manual_seed(0); // For reproducible results
//
//     // Dropout probability
//     double dropout_rate = 0.5;
//     AttentionDropout att_dropout_module(dropout_rate);
//     std::cout << "Attention Dropout Module: " << att_dropout_module << std::endl;
//
//     // Example input (e.g., attention scores or attention output)
//     torch::Tensor input_tensor = torch::ones({2, 3, 4}); // Batch_size=2, Seq_len=3, Dim=4
//     std::cout << "Input Tensor:\n" << input_tensor << std::endl;
//
//     // --- Training mode ---
//     att_dropout_module->train(); // Set the module to training mode
//     torch::Tensor output_train = att_dropout_module->forward(input_tensor);
//     std::cout << "Output (training mode):\n" << output_train << std::endl;
//     // Expected non-zero elements to be scaled by 1 / (1 - 0.5) = 2
//     // Approximately 50% of elements should be zero.
//
//     // --- Evaluation mode ---
//     att_dropout_module->eval(); // Set the module to evaluation mode
//     torch::Tensor output_eval = att_dropout_module->forward(input_tensor);
//     std::cout << "Output (evaluation mode):\n" << output_eval << std::endl;
//     // Expected output to be identical to input in evaluation mode.
//
//     // --- Dropout probability 0 ---
//     AttentionDropout no_dropout_module(0.0);
//     no_dropout_module->train();
//     torch::Tensor output_no_dropout = no_dropout_module->forward(input_tensor);
//     std::cout << "Output (p=0.0, training mode):\n" << output_no_dropout << std::endl;
//     // Expected output to be identical to input.
//
//     // --- Dropout probability 1 ---
//     AttentionDropout full_dropout_module(1.0);
//     full_dropout_module->train();
//     torch::Tensor output_full_dropout = full_dropout_module->forward(input_tensor);
//     std::cout << "Output (p=1.0, training mode):\n" << output_full_dropout << std::endl;
//     // Expected output to be all zeros.
// }
//
// // int main() {
// //    run_attention_dropout_example();
// //    return 0;
// // }
// */


namespace xt::dropouts
{
    AttentionDropout::AttentionDropout(double p) : p_(p)
    {
        TORCH_CHECK(p_ >= 0.0 && p_ <= 1.0, "Dropout probability has to be between 0 and 1, but got ", p_);
    }

    auto AttentionDropout::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        vector<std::any> tensors_ = tensors;
        auto input = std::any_cast<torch::Tensor>(tensors_[0]);


        if (!this->is_training() || p_ == 0.0) {
            return input;
        }

        // If p_ is 1.0, all elements are dropped.
        if (p_ == 1.0) {
            return torch::zeros_like(input);
        }

        // Create a mask where elements are 1 with probability (1.0 - p_)
        // and 0 with probability p_.
        // Then, scale the result by 1.0 / (1.0 - p_).
        // This is the "inverted dropout" technique.
        torch::Tensor mask = (torch::rand_like(input) < (1.0 - p_)).to(input.dtype());

        // Ensure denominator is not zero (already handled by p_ == 1.0 check, but good for clarity)
        return (input * mask) / (1.0 - p_);


    }
}
