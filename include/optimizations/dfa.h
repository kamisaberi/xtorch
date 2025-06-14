#ifndef DFA_OPTIMIZER_HPP
#define DFA_OPTIMIZER_HPP

#include <torch/torch.h>
#include <vector>
#include <memory>
#include <functional>
#include <unordered_map>

// --- Options for DFA ---
struct DFAOptions {
    double learning_rate = 0.01;
};

// --- Per-layer state for DFA (simpler now) ---
struct DFALayerState {
    // A raw pointer is fine here since the model owns the module and will outlive the optimizer
    torch::nn::LinearImpl* module_impl;
    torch::Tensor feedback_matrix; // The fixed, random feedback matrix 'B'
};

// --- DFA Optimizer Class (Meta-Optimizer) ---
class DFA {
public:
    DFA(std::shared_ptr<torch::nn::Module> model, DFAOptions options);

    void step(const torch::Tensor& input, const torch::Tensor& target,
              const std::function<torch::Tensor(torch::Tensor, torch::Tensor)>& loss_fn_derivative);

    void zero_grad() {} // No-op

private:
    std::shared_ptr<torch::nn::Module> model_;
    DFAOptions options_;

    // Store state for hidden layers
    std::vector<DFALayerState> dfa_hidden_states_;

    // Store a direct pointer to the final layer implementation
    torch::nn::LinearImpl* final_layer_impl_ = nullptr;
};

#endif // DFA_OPTIMIZER_HPP