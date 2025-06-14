#include "include/optimizations/dfa.h"
#include <stdexcept>

// --- DFA Implementation ---

DFA::DFA(std::shared_ptr<torch::nn::Module> model, DFAOptions options)
    : model_(model), options_(options) {

    TORCH_CHECK(model_ != nullptr, "A valid model must be provided.");
    torch::NoGradGuard no_grad;

    auto sequential_model = model_->as<torch::nn::Sequential>();
    TORCH_CHECK(sequential_model != nullptr, "DFA implementation requires a torch::nn::Sequential model.");

    // Find the last linear layer to determine the final output dimension
    torch::nn::Linear last_linear_module = nullptr;
    for (auto it = sequential_model->children().rbegin(); it != sequential_model->children().rend(); ++it) {
        if (auto linear_layer = (*it)->as<torch::nn::Linear>()) {
            last_linear_module = linear_layer;
            break;
        }
    }
    TORCH_CHECK(last_linear_module, "Could not find any Linear layer in the model.");
    auto final_output_dim = last_linear_module->options.out_features();

    // Now, iterate through all modules to set up state
    for (const auto& module : sequential_model->children()) {
        if (auto linear_layer = module->as<torch::nn::Linear>()) {
            // Check if this is the final linear layer we found
            if (linear_layer.get() == last_linear_module.get()) {
                final_layer_impl_ = linear_layer.get();
            } else {
                // This is a hidden layer
                DFALayerState state;
                state.module_impl = linear_layer.get();
                auto output_dim = linear_layer->options.out_features();

                // B projects from final_output_dim to this layer's output_dim
                state.feedback_matrix = torch::randn({final_output_dim, output_dim}, torch::kFloat32);
                dfa_hidden_states_.push_back(std::move(state));
            }
        }
    }
    TORCH_CHECK(final_layer_impl_ != nullptr, "Final linear layer was found but could not be assigned.");
}

void DFA::step(const torch::Tensor& input, const torch::Tensor& target,
               const std::function<torch::Tensor(torch::Tensor, torch::Tensor)>& loss_fn_derivative) {

    torch::NoGradGuard no_grad;

    // --- 1. Custom Forward Pass to store activations for each layer ---
    std::unordered_map<torch::nn::ModuleImpl*, torch::Tensor> layer_inputs;

    torch::Tensor current_tensor = input;
    auto sequential_model = model_->as<torch::nn::Sequential>();

    for (const auto& module : sequential_model->children()) {
        // Store the input *before* passing it through the module
        if (auto linear_layer = module->as<torch::nn::Linear>()) {
            layer_inputs[linear_layer.get()] = current_tensor.clone();
        }
        current_tensor = module->forward(current_tensor);
    }
    auto final_output = current_tensor;

    // --- 2. Calculate Error at the Output Layer ---
    auto error_output = loss_fn_derivative(final_output, target);

    // --- 3. Update the Final Layer using the true error ---
    if (final_layer_impl_) {
        // Retrieve the saved input for the final layer
        auto input_to_final_layer = layer_inputs.at(final_layer_impl_);

        auto delta_W_final = -options_.learning_rate * input_to_final_layer.t().matmul(error_output);
        final_layer_impl_->weight.add_(delta_W_final.t());
        if (final_layer_impl_->bias.defined()) {
            final_layer_impl_->bias.add_(-options_.learning_rate * error_output.sum(0));
        }
    }

    // --- 4. Update Hidden Layers using Direct Feedback Alignment ---
    for (auto& state : dfa_hidden_states_) {
        // Project the output error back using the fixed random matrix B
        auto error_hidden = error_output.matmul(state.feedback_matrix);

        // This is where one would multiply by the derivative of the activation function, f'(a_h).
        // For simplicity (e.g., assuming ReLU where derivative is 0 or 1), we omit this.

        // Retrieve the saved input for this hidden layer
        auto input_activation = layer_inputs.at(state.module_impl);

        // Calculate weight update: delta_W = -lr * a_in^T * error_hidden
        auto delta_W = -options_.learning_rate * input_activation.t().matmul(error_hidden);

        // Apply the update directly to the implementation's parameters
        state.module_impl->weight.add_(delta_W.t());
        if (state.module_impl->bias.defined()) {
             state.module_impl->bias.add_(-options_.learning_rate * error_hidden.sum(0));
        }
    }
}