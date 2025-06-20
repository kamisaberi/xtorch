#pragma once


#include "common.h"
// --- Options for AdaMod Optimizer ---
struct AdaModOptions : torch::optim::OptimizerOptions {
    explicit AdaModOptions(double learning_rate = 1e-3)
        : torch::optim::OptimizerOptions() {
        this->lr(learning_rate);
    }

    TORCH_ARG(double, beta1) = 0.9;
    TORCH_ARG(double, beta2) = 0.999;
    TORCH_ARG(double, beta3) = 0.999; // Decay for the optional EMA of the denominator
    TORCH_ARG(double, eps) = 1e-8;
    TORCH_ARG(double, weight_decay) = 0.0;
    TORCH_ARG(double ,  lr) = 1e-6;

    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<torch::optim::OptimizerOptions> clone() const override;
};

// --- Parameter State for AdaMod Optimizer ---
struct AdaModParamState : torch::optim::OptimizerParamState {
    TORCH_ARG(torch::Tensor, step);
    TORCH_ARG(torch::Tensor, exp_avg);      // m_t
    TORCH_ARG(torch::Tensor, exp_avg_sq);   // v_t
    TORCH_ARG(torch::Tensor, long_term_avg_sq); // s_t, the long-term memory

    // AdaModParamState() = default;
    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<OptimizerParamState> clone() const override;
};

// --- AdaMod Optimizer Class ---
class AdaMod : public torch::optim::Optimizer {
public:
    AdaMod(std::vector<torch::Tensor> params, AdaModOptions options);
    explicit AdaMod(std::vector<torch::Tensor> params, double lr = 1e-3);

    using LossClosure = std::function<torch::Tensor()>;
    torch::Tensor step(LossClosure closure = nullptr) override;
    void save(torch::serialize::OutputArchive& archive) const override;
    void load(torch::serialize::InputArchive& archive) override;

protected:
    std::unique_ptr<torch::optim::OptimizerParamState> make_param_state() ;
};

