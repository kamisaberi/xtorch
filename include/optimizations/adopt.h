#ifndef ADOPT_OPTIMIZER_HPP
#define ADOPT_OPTIMIZER_HPP

#include <torch/torch.h>
#include <vector>
#include <memory>
#include <functional>

// --- Options for AdOpt Optimizer ---
struct AdOptOptions {
    // AdOpt is mostly self-tuning. It has a base LR for decoupled weight decay.
    double lr = 1.0;
    double weight_decay = 1e-4;
    double eps = 1e-8;
    // EMA decay for smoothing statistics
    double beta1 = 0.9;
    double beta2 = 0.999;
};

// --- Per-parameter state for AdOpt ---
// AdOpt's state is global, but we can store it in the first param's state.
// Here, we create per-param state for simplicity of the framework.
struct AdOptParamState : torch::optim::OptimizerParamState {
    TORCH_ARG(torch::Tensor, step);
    TORCH_ARG(torch::Tensor, z);            // Dual variable
    TORCH_ARG(torch::Tensor, hess_diag_ema); // Smoothed Hessian diagonal (preconditioner H_t)
    TORCH_ARG(torch::Tensor, grad_ema);      // Smoothed gradient (for smoothness estimation)
    TORCH_ARG(torch::Tensor, grad_sq_ema);   // Smoothed squared gradient

    // AdOptParamState() = default;
    std::unique_ptr<OptimizerParamState> clone() const override {
        auto cloned = std::make_unique<AdOptParamState>();
        if(step().defined()) cloned->step(step().clone());
        if(z().defined()) cloned->z(z().clone());
        if(hess_diag_ema().defined()) cloned->hess_diag_ema(hess_diag_ema().clone());
        if(grad_ema().defined()) cloned->grad_ema(grad_ema().clone());
        if(grad_sq_ema().defined()) cloned->grad_sq_ema(grad_sq_ema().clone());
        return cloned;
    }
};

// --- AdOpt Optimizer Class (Meta-Optimizer) ---
class AdOpt {
public:
    AdOpt(std::vector<torch::Tensor> params, AdOptOptions options);

    void step(torch::Tensor& loss);
    void zero_grad();

private:
    std::vector<torch::Tensor> params_;
    std::unordered_map<c10::TensorImpl*, std::unique_ptr<AdOptParamState>> state_;
    AdOptOptions options_;

    void _init_state(torch::Tensor& p);
};

#endif // ADOPT_OPTIMIZER_HPP