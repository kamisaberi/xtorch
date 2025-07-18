#include <dropouts/drop_connect.h>


//
// #include <torch/torch.h>
// #include <ostream> // For std::ostream
//
// struct DropConnectImpl : torch::nn::Module {
//     double p_drop_;
//     double epsilon_ = 1e-7;
//
//     DropConnectImpl(double p_drop = 0.5) : p_drop_(p_drop) {
//         TORCH_CHECK(p_drop_ >= 0.0 && p_drop_ <= 1.0, "DropConnect probability p_drop must be between 0 and 1.");
//     }
//
//     torch::Tensor forward(const torch::Tensor& weight_tensor) {
//         if (!this->is_training() || p_drop_ == 0.0) {
//             return weight_tensor;
//         }
//
//         if (p_drop_ == 1.0) {
//             return torch::zeros_like(weight_tensor);
//         }
//
//         double keep_prob = 1.0 - p_drop_;
//         torch::Tensor mask = torch::bernoulli(
//             torch::full_like(weight_tensor, keep_prob)
//         ).to(weight_tensor.dtype());
//
//         return (weight_tensor * mask) / (keep_prob + epsilon_);
//     }
//
//     void pretty_print(std::ostream& stream) const override {
//         stream << "DropConnect(p_drop=" << p_drop_ << ")";
//     }
// };
//
// TORCH_MODULE(DropConnect);
//
// /*
// // --- Example: How DropConnect might be used within a custom Linear layer ---
// // This is for context and not part of the DropConnectImpl module itself.
//
// struct CustomLinearWithDropConnectImpl : torch::nn::Module {
//     torch::Tensor weight;
//     torch::Tensor bias;
//     DropConnect drop_connect_module; // Instance of DropConnect
//
//     CustomLinearWithDropConnectImpl(int64_t in_features, int64_t out_features, double dc_p_drop = 0.2)
//         : drop_connect_module(dc_p_drop) { // Initialize the DropConnect module
//         weight = register_parameter("weight", torch::randn({out_features, in_features}));
//         bias = register_parameter("bias", torch::randn({out_features}));
//     }
//
//     torch::Tensor forward(const torch::Tensor& input) {
//         torch::Tensor current_weight = weight; // Start with the original weight
//
//         if (this->is_training()) {
//             // Apply DropConnect to a copy of the weights during training
//             current_weight = drop_connect_module->forward(weight);
//             // Optionally, DropConnect (or standard Dropout) could also be applied to the bias.
//         }
//         // During evaluation, current_weight remains the original 'this->weight'.
//
//         return torch::nn::functional::linear(input, current_weight, bias);
//     }
// };
// TORCH_MODULE(CustomLinearWithDropConnect);
//
//
// #include <iostream>
// void run_example_usage() {
//     torch::manual_seed(0);
//
//     // Using DropConnect module directly on a tensor
//     DropConnect dc_op(0.5);
//     dc_op->train(); // Set to training mode
//     torch::Tensor W = torch::ones({2,3});
//     std::cout << "Original W:\n" << W << std::endl;
//     torch::Tensor W_dropped = dc_op->forward(W);
//     std::cout << "W after DropConnect (train):\n" << W_dropped << std::endl;
//
//     dc_op->eval(); // Set to evaluation mode
//     W_dropped = dc_op->forward(W);
//     std::cout << "W after DropConnect (eval):\n" << W_dropped << std::endl;
//
//     // Using the custom layer that incorporates DropConnect
//     CustomLinearWithDropConnect custom_layer(3, 2, 0.3);
//     torch::Tensor layer_input = torch::randn({4, 3});
//
//     custom_layer->train();
//     torch::Tensor layer_output_train = custom_layer->forward(layer_input);
//     std::cout << "\nCustomLayer output (train):\n" << layer_output_train << std::endl;
//     // (Internally, the weights used for linear operation were DropConnected)
//
//     custom_layer->eval();
//     torch::Tensor layer_output_eval = custom_layer->forward(layer_input);
//     std::cout << "CustomLayer output (eval):\n" << layer_output_eval << std::endl;
//     // (Internally, the original weights were used)
// }
// // int main() {
// //    run_example_usage();
// //    return 0;
// // }
// */


namespace xt::dropouts
{
    torch::Tensor drop_connect(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    DropConnect::DropConnect(double p_drop ) : p_drop_(p_drop)
    {
        TORCH_CHECK(p_drop_ >= 0.0 && p_drop_ <= 1.0, "DropConnect probability p_drop must be between 0 and 1.");
    }

    auto DropConnect::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        vector<std::any> tensors_ = tensors;
        auto weight_tensor = std::any_cast<torch::Tensor>(tensors_[0]);

        if (!this->is_training() || p_drop_ == 0.0)
        {
            return weight_tensor;
        }

        if (p_drop_ == 1.0)
        {
            return torch::zeros_like(weight_tensor);
        }

        double keep_prob = 1.0 - p_drop_;
        torch::Tensor mask = torch::bernoulli(
            torch::full_like(weight_tensor, keep_prob)
        ).to(weight_tensor.dtype());

        return (weight_tensor * mask) / (keep_prob + epsilon_);
    }
}
