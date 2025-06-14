#ifndef APOLLO_OPTIMIZER_HPP
#define APOLLO_OPTIMIZER_HPP

#include <torch/torch.h>
#include <vector>
#include <memory>
#include <functional>

// --- Options for Apollo Optimizer ---
struct ApolloOptions {
    double lr = 1e-2; // Apollo can often use a higher base LR
    double beta1 = 0.9;
    double beta2 = 0.999;
    double eps = 1e-8;
    double weight_decay = 1e-4;

    ApolloOptions& learning_rate(double new_lr) { lr = new_lr; return *this; }
    ApolloOptions& betas(std::tuple<double, double> new_betas) {
        beta1 = std::get<0>(new_betas);
        beta2 = std::get<1>(new_betas);
        return *this;
    }
};

// --- Per-parameter state for Apollo ---
struct ApolloParamState : torch::optim::OptimizerParamState {
    TORCH_ARG(torch::Tensor, step);
    TORCH_ARG(torch::Tensor, exp_avg);      // m_t
    TORCH_ARG(torch::Tensor, hess_diag_ema); // v_t (EMA of Hessian diagonal)
    TORCH_ARG(torch::Tensor, bias_rectifier); // B_t (running max of v_hat)

    ApolloParamState() = default;
    std::unique_ptr<OptimizerParamState> clone() const override {
        auto cloned = std::make_unique<ApolloParamState>();
        if(step().defined()) cloned->step(step().clone());
        if(exp_avg().defined()) cloned->exp_avg(exp_avg().clone());
        if(hess_diag_ema().defined()) cloned->hess_diag_ema(hess_diag_ema().clone());
        if(bias_rectifier().defined()) cloned->bias_rectifier(bias_rectifier().clone());
        return cloned;
    }
};

// --- Apollo Optimizer Class (Meta-Optimizer) ---
class Apollo {
public:
    Apollo(std::vector<torch::Tensor> params, ApolloOptions options);

    void step(torch::Tensor& loss);
    void zero_grad();

private:
    std::vector<torch::Tensor> params_;
    std::unordered_map<c10::TensorImpl*, std::unique_ptr<ApolloParamState>> state_;
    ApolloOptions options_;

    void _init_state(torch::Tensor& p);
};

#endif // APOLLO_OPTIMIZER_HPP