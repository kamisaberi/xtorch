// #include "include/optimizations/dfa.h"
// #include <stdexcept>
//
// // --- DFA Implementation ---
//
// DFA::DFA(std::shared_ptr<torch::nn::Module> model, DFAOptions options)
//     : model_(model), options_(options) {
//
//     TORCH_CHECK(model_ != nullptr, "A valid model must be provided.");
//     torch::NoGradGuard no_grad;
//
//     auto sequential_model = model_->as<torch::nn::Sequential>();
//     TORCH_CHECK(sequential_model != nullptr, "DFA implementation requires a torch::nn::Sequential model.");
//
//     // --- Correctly Find and Store the Final Layer Handle ---
//     // Iterate backwards to find the last Linear module in the sequence
//     for (auto it = sequential_model->children().rbegin(); it != sequential_model->children().rend(); ++it) {
//         if (auto linear_layer = (*it)->as<torch::nn::Linear>()) {
//             this->final_layer_ = linear_layer;
//             break;
//         }
//     }
//     TORCH_CHECK(this->final_layer_, "Could not find any Linear layer in the model.");
//     auto final_output_dim = this->final_layer_->options.out_features();
//
//     // --- Correctly Iterate and Set Up State ---
//     for (const auto& module_handle : sequential_model->children()) {
//         if (auto linear_layer = module_handle->as<torch::nn::Linear>()) {
//             // Compare the implementation pointers to see if it's the final layer
//             if (linear_layer.get() != this->final_layer_.get()) {
//                 // This is a hidden layer. Create its state.
//                 DFALayerState state;
//                 state.module = linear_layer; // Store the handle
//                 auto output_dim = linear_layer->options.out_features();
//
//                 state.feedback_matrix = torch::randn({final_output_dim, output_dim}, torch::kFloat32);
//                 dfa_hidden_layers_.push_back(std::move(state));
//             }
//         }
//     }
// }
//
// void DFA::step(const torch::Tensor& input, const torch::Tensor& target,
//                const std::function<torch::Tensor(torch::Tensor, torch::Tensor)>& loss_fn_derivative) {
//
//     torch::NoGradGuard no_grad;
//
//     // --- 1. Custom Forward Pass to store activations for each layer ---
//     std::vector<torch::Tensor> hidden_layer_inputs;
//     torch::Tensor final_layer_input;
//
//     torch::Tensor current_tensor = input;
//     auto sequential_model = model_->as<torch::nn::Sequential>();
//
//     for (const auto& module_handle : sequential_model->children()) {
//         if (auto linear_layer = module_handle->as<torch::nn::Linear>()) {
//             // Check if it's a hidden or final layer and store its input
//             if (linear_layer.get() == final_layer_.get()) {
//                 final_layer_input = current_tensor.clone();
//             } else {
//                 hidden_layer_inputs.push_back(current_tensor.clone());
//             }
//         }
//         current_tensor = module_handle->forward(current_tensor);
//     }
//     auto final_output = current_tensor;
//
//     // Check if we captured the right number of inputs
//     TORCH_CHECK(hidden_layer_inputs.size() == dfa_hidden_layers_.size(), "Mismatch between hidden layers and captured inputs.");
//
//     // --- 2. Calculate Error at the Output Layer ---
//     auto error_output = loss_fn_derivative(final_output, target);
//
//     // --- 3. Update the Final Layer using the true error ---
//     if (final_layer_ && final_layer_input.defined()) {
//         auto delta_W_final = -options_.learning_rate * final_layer_input.t().matmul(error_output);
//         final_layer_->weight.add_(delta_W_final.t());
//         if (final_layer_->bias.defined()) {
//             final_layer_->bias.add_(-options_.learning_rate * error_output.sum(0));
//         }
//     }
//
//     // --- 4. Update Hidden Layers using Direct Feedback Alignment ---
//     for (size_t i = 0; i < dfa_hidden_layers_.size(); ++i) {
//         auto& state = dfa_hidden_layers_[i];
//         auto& input_activation = hidden_layer_inputs[i];
//
//         // Project the output error back using the fixed random matrix B
//         auto error_hidden = error_output.matmul(state.feedback_matrix);
//
//         // Calculate weight update
//         auto delta_W = -options_.learning_rate * input_activation.t().matmul(error_hidden);
//
//         // Apply the update using the stored module handle
//         state.module->weight.add_(delta_W.t());
//         if (state.module->bias.defined()) {
//              state.module->bias.add_(-options_.learning_rate * error_hidden.sum(0));
//         }
//     }
// }