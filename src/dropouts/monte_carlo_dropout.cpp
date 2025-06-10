#include "include/dropouts/monte_carlo_dropout.h"


// #include <torch/torch.h>
// #include <ostream> // For std::ostream
//
// // This is essentially a standard dropout layer.
// // The "Monte Carlo" aspect refers to its usage during inference (dropout enabled).
// struct MonteCarloDropoutImpl : torch::nn::Module {
//     double p_drop_; // Probability of an element to be zeroed out.
//     double epsilon_ = 1e-7; // For numerical stability in division
//
//     MonteCarloDropoutImpl(double p_drop = 0.5) : p_drop_(p_drop) {
//         TORCH_CHECK(p_drop_ >= 0.0 && p_drop_ <= 1.0, "Dropout probability p_drop must be between 0 and 1.");
//     }
//
//     // The key for MC Dropout is that this forward pass (with dropout logic)
//     // might be called even when model->is_training() is false, if explicitly
//     // intended for MC sampling. However, typical `torch::nn::Module` practice
//     // is to disable dropout if !this->is_training().
//     //
//     // For MC Dropout, you typically call `model->train()` before inference passes
//     // to ensure dropout layers are active, or have a separate flag.
//     // This implementation will follow standard behavior: dropout only active if this->is_training().
//     // The user must call `module->train()` before MC inference passes.
//     torch::Tensor forward(const torch::Tensor& input) {
//         if (!this->is_training() || p_drop_ == 0.0) {
//             // If not in "training mode" (dropout active) or if p_drop is zero,
//             // return the input as is.
//             return input;
//         }
//
//         if (p_drop_ == 1.0) {
//             // If dropout probability is one, all elements are zeroed out.
//             return torch::zeros_like(input);
//         }
//
//         double keep_prob = 1.0 - p_drop_;
//         torch::Tensor mask = torch::bernoulli(
//             torch::full_like(input, keep_prob)
//         ).to(input.dtype());
//
//         return (input * mask) / (keep_prob + epsilon_);
//     }
//
//     void pretty_print(std::ostream& stream) const override {
//         stream << "MonteCarloDropout(p_drop=" << p_drop_ << ")";
//     }
// };
//
// TORCH_MODULE(MonteCarloDropout); // Creates the MonteCarloDropout module "class"
//
// /*
// // Example of how to use a model with MonteCarloDropout for MC inference:
// // (This is for illustration and not part of the MonteCarloDropoutImpl itself)
//
// #include <vector>
// #include <iostream>
//
// // A simple model that uses our MonteCarloDropout layer
// struct SimpleModelForMCDropout : torch::nn::Module {
//     torch::nn::Linear fc1{nullptr};
//     MonteCarloDropout mc_dropout_layer{nullptr}; // Using our dropout layer
//     torch::nn::Linear fc2{nullptr};
//
//     SimpleModelForMCDropout(int64_t in_dim, int64_t hidden_dim, int64_t out_dim, double dropout_p = 0.25) {
//         fc1 = register_module("fc1", torch::nn::Linear(in_dim, hidden_dim));
//         mc_dropout_layer = register_module("mc_dropout", MonteCarloDropout(dropout_p));
//         fc2 = register_module("fc2", torch::nn::Linear(hidden_dim, out_dim));
//     }
//
//     torch::Tensor forward(const torch::Tensor& x) {
//         auto out = torch::relu(fc1(x));
//         out = mc_dropout_layer(out); // Dropout is applied here
//         out = fc2(out);
//         return out;
//     }
// };
// TORCH_MODULE(SimpleModelForMCDropout);
//
//
// void run_mc_dropout_example() {
//     torch::manual_seed(0); // For reproducible results
//
//     int in_features = 10;
//     int hidden_features = 20;
//     int out_features = 1; // e.g., regression output
//     double dropout_rate = 0.5;
//
//     SimpleModelForMCDropout model(in_features, hidden_features, out_features, dropout_rate);
//     std::cout << "Model with MonteCarloDropout: " << model << std::endl;
//
//     torch::Tensor input_sample = torch::randn({1, in_features}); // A single input sample
//
//     // --- Standard Training (example, not actually training here) ---
//     // model->train();
//     // ... training loop ...
//
//     // --- Standard Evaluation (dropout OFF) ---
//     model->eval(); // This turns OFF dropout in mc_dropout_layer
//     torch::Tensor prediction_eval_mode = model->forward(input_sample);
//     std::cout << "\n--- Standard Evaluation (Dropout OFF) ---" << std::endl;
//     std::cout << "Prediction (eval mode): " << prediction_eval_mode << std::endl;
//     // Prediction will be deterministic because dropout is off.
//
//     // --- Monte Carlo Inference (Dropout ON) ---
//     std::cout << "\n--- Monte Carlo Inference (Dropout ON) ---" << std::endl;
//     model->train(); // CRITICAL: Set model to training mode to activate dropout layers for MC sampling.
//                     // This is how you enable dropout during inference for MC Dropout.
//
//     int num_mc_samples = 10;
//     std::vector<torch::Tensor> mc_predictions;
//     mc_predictions.reserve(num_mc_samples);
//
//     std::cout << "MC Predictions (dropout active):" << std::endl;
//     for (int i = 0; i < num_mc_samples; ++i) {
//         torch::Tensor prediction_mc_pass = model->forward(input_sample);
//         mc_predictions.push_back(prediction_mc_pass);
//         std::cout << "MC Pass " << i << ": " << prediction_mc_pass.item<float>() << std::endl;
//         // These predictions will vary due to active dropout.
//     }
//
//     // Stack predictions to analyze
//     if (!mc_predictions.empty()) {
//         torch::Tensor all_mc_predictions = torch::stack(mc_predictions); // Shape: [num_mc_samples, 1, out_features]
//
//         // Calculate mean and variance (or std) for uncertainty estimation
//         torch::Tensor mean_prediction = all_mc_predictions.mean({0}); // Mean over MC samples
//         torch::Tensor variance_prediction = all_mc_predictions.var({0}); // Variance over MC samples
//         // For multi-dimensional output, you might want variance per output dimension.
//
//         std::cout << "\nMC Results:" << std::endl;
//         std::cout << "Mean Prediction: " << mean_prediction << std::endl;
//         std::cout << "Variance of Predictions (Uncertainty): " << variance_prediction << std::endl;
//     }
//
//     // Remember to set model back to eval() if doing further standard evaluation
//     // model->eval();
// }
//
// // int main() {
// //    run_mc_dropout_example();
// //    return 0;
// // }
// */



namespace xt::dropouts
{
    torch::Tensor monte_carlo_dropout(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto MonteCarloDropout::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::dropouts::monte_carlo_dropout(torch::zeros(10));
    }
}
