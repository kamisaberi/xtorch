// #ifndef DFA_OPTIMIZER_HPP
// #define DFA_OPTIMIZER_HPP
//
// #include <torch/torch.h>
// #include <vector>
// #include <memory>
// #include <functional>
//
// // --- Options for DFA ---
// struct DFAOptions {
//     double learning_rate = 0.01;
// };
//
// // --- Per-layer state for DFA hidden layers ---
// struct DFALayerState {
//     torch::nn::Linear module; // Store the module handle directly
//     torch::Tensor feedback_matrix;
// };
//
// // --- DFA Optimizer Class (Meta-Optimizer) ---
// class DFA {
// public:
//     DFA(std::shared_ptr<torch::nn::Module> model, DFAOptions options);
//
//     void step(const torch::Tensor& input, const torch::Tensor& target,
//               const std::function<torch::Tensor(torch::Tensor, torch::Tensor)>& loss_fn_derivative);
//
// private:
//     std::shared_ptr<torch::nn::Module> model_;
//     DFAOptions options_;
//
//     // Store handles to the hidden layers and their state
//     std::vector<DFALayerState> dfa_hidden_layers_;
//
//     // Store a direct handle to the final layer
//     torch::nn::Linear final_layer_ = nullptr;
// };
//
// #endif // DFA_OPTIMIZER_HPP