
#pragma once


#include "common.h"
// --- Options for AdaBound Optimizer (Corrected) ---
struct AdaBoundOptions : torch::optim::OptimizerOptions {
    explicit AdaBoundOptions(double learning_rate = 1e-3)
        : torch::optim::OptimizerOptions() {
        this->lr(learning_rate);
    }

    TORCH_ARG(double, beta1) = 0.9;
    TORCH_ARG(double, beta2) = 0.999;
    TORCH_ARG(double, final_lr) = 0.1;
    TORCH_ARG(double, gamma) = 1e-3;
    TORCH_ARG(double, eps) = 1e-8;
    TORCH_ARG(double, weight_decay) = 0.0;
    TORCH_ARG(double ,  lr) = 1e-6;

    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<torch::optim::OptimizerOptions> clone() const override;
};

// --- Parameter State for AdaBound Optimizer (Correct) ---
struct AdaBoundParamState : torch::optim::OptimizerParamState {
    TORCH_ARG(torch::Tensor, step);
    TORCH_ARG(torch::Tensor, exp_avg);
    TORCH_ARG(torch::Tensor, exp_avg_sq);

    // AdaBoundParamState() = default;
    std::unique_ptr<OptimizerParamState> clone() const override {
        auto cloned = std::make_unique<AdaBoundParamState>();
        if(step().defined()) cloned->step(step().clone());
        if(exp_avg().defined()) cloned->exp_avg(exp_avg().clone());
        if(exp_avg_sq().defined()) cloned->exp_avg_sq(exp_avg_sq().clone());
        return cloned;
    }
};

// --- AdaBound Optimizer Class (Declarations Added) ---
class AdaBound : public torch::optim::Optimizer {
public:
    AdaBound(std::vector<torch::Tensor> params, AdaBoundOptions options);
    explicit AdaBound(std::vector<torch::Tensor> params, double lr = 1e-3);

    using LossClosure = std::function<torch::Tensor()>;
    torch::Tensor step(LossClosure closure = nullptr) override;

    // --- ADDED DECLARATIONS FOR SAVE AND LOAD ---
    void save(torch::serialize::OutputArchive& archive) const override;
    void load(torch::serialize::InputArchive& archive) override;

protected:
    std::unique_ptr<torch::optim::OptimizerParamState> make_param_state() ;
};
