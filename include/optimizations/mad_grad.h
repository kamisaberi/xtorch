#pragma once


#include "common.h"

// --- Options for MADGRAD Optimizer ---
struct MADGRADOptions : torch::optim::OptimizerOptions {
    explicit MADGRADOptions(double learning_rate = 1e-2) // MADGRAD often works well with slightly higher LR
        : torch::optim::OptimizerOptions() {
        this->lr(learning_rate);
    }

    // Momentum decay (lambda in the paper)
    TORCH_ARG(double, momentum) = 0.9;
    TORCH_ARG(double, weight_decay) = 1e-4;
    TORCH_ARG(double, eps) = 1e-6; // Epsilon for numerical stability
    TORCH_ARG(double, lr) = 1e-6; // Epsilon for numerical stability

    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<torch::optim::OptimizerOptions> clone() const override;
};

// --- Parameter State for MADGRAD Optimizer ---
struct MADGRADParamState : torch::optim::OptimizerParamState {
    TORCH_ARG(torch::Tensor, step);
    TORCH_ARG(torch::Tensor, grad_sum);       // s_k in the paper (sum of gradients)
    TORCH_ARG(torch::Tensor, grad_sum_sq);    // v_k in the paper (sum of squared grads + volatility)
    TORCH_ARG(torch::Tensor, grad_prev);      // g_{k-1} to calculate volatility

    // MADGRADParamState() = default;
    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<OptimizerParamState> clone() const override;
};

// --- MADGRAD Optimizer Class ---
class MADGRAD : public torch::optim::Optimizer {
public:
    MADGRAD(std::vector<torch::Tensor> params, MADGRADOptions options);
    explicit MADGRAD(std::vector<torch::Tensor> params, double lr = 1e-2);

    using LossClosure = std::function<torch::Tensor()>;
    torch::Tensor step(LossClosure closure = nullptr) override;
    void save(torch::serialize::OutputArchive& archive) const override;
    void load(torch::serialize::InputArchive& archive) override;

protected:
    std::unique_ptr<torch::optim::OptimizerParamState> make_param_state() ;
};

