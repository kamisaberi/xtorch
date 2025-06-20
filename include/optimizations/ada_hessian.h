#pragma once


#include "common.h"
// --- Options for AdaHessian Optimizer ---
struct AdaHessianOptions {
public:
    // We use a custom struct because this is a meta-optimizer
    double lr = 0.1;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double eps = 1e-8;
    double weight_decay = 1e-4;

    AdaHessianOptions& learning_rate(double new_lr) { lr = new_lr; return *this; }
    AdaHessianOptions& betas(std::tuple<double, double> new_betas) {
        beta1 = std::get<0>(new_betas);
        beta2 = std::get<1>(new_betas);
        return *this;
    }
};

// --- Per-parameter state for AdaHessian ---
struct AdaHessianParamState :public torch::optim::OptimizerParamState {
    TORCH_ARG(torch::Tensor, step);
    TORCH_ARG(torch::Tensor, exp_avg);      // m_t
    TORCH_ARG(torch::Tensor, exp_avg_sq);   // v_t (EMA of Hessian diagonal squared)
public:
    // AdaHessianParamState() = default;
    std::unique_ptr<OptimizerParamState> clone() const override {
        auto cloned = std::make_unique<AdaHessianParamState>();
        if(step().defined()) cloned->step(step().clone());
        if(exp_avg().defined()) cloned->exp_avg(exp_avg().clone());
        if(exp_avg_sq().defined()) cloned->exp_avg_sq(exp_avg_sq().clone());
        return cloned;
    }
};

// --- AdaHessian Optimizer Class (Meta-Optimizer) ---
class AdaHessian {
public:
    AdaHessian(std::vector<torch::Tensor> params, AdaHessianOptions options);

    // The step function needs the loss tensor to trigger the backward passes.
    void step(torch::Tensor& loss);
    void zero_grad();

private:
    std::vector<torch::Tensor> params_;
    std::unordered_map<c10::TensorImpl*, std::unique_ptr<AdaHessianParamState>> state_;
    AdaHessianOptions options_;

    // Helper to initialize state for a parameter
    void _init_state(torch::Tensor& p);
};

