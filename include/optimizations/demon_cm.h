#pragma once


#include "common.h"

// --- Options for DemonCM Optimizer ---
struct DemonCMOptions : torch::optim::OptimizerOptions {
    explicit DemonCMOptions(double learning_rate = 0.1)
        : torch::optim::OptimizerOptions() {
        this->lr(learning_rate);
    }

    // Decaying momentum parameters
    TORCH_ARG(double, beta_initial) = 0.99;
    TORCH_ARG(double, beta_final) = 0.5;
    TORCH_ARG(long, total_steps) = 100000;

    TORCH_ARG(double, weight_decay) = 1e-4;
    TORCH_ARG(double ,  lr) = 1e-6;

    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<torch::optim::OptimizerOptions> clone() const override;
};

// --- Parameter State for DemonCM Optimizer ---
struct DemonCMParamState : torch::optim::OptimizerParamState {
    TORCH_ARG(torch::Tensor, step);
    TORCH_ARG(torch::Tensor, momentum_buffer);

    // DemonCMParamState() = default;
    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<OptimizerParamState> clone() const override;
};

// --- DemonCM Optimizer Class ---
class DemonCM : public torch::optim::Optimizer {
public:
    DemonCM(std::vector<torch::Tensor> params, DemonCMOptions options);
    explicit DemonCM(std::vector<torch::Tensor> params, double lr = 0.1);

    using LossClosure = std::function<torch::Tensor()>;
    torch::Tensor step(LossClosure closure = nullptr) override;
    void save(torch::serialize::OutputArchive& archive) const override;
    void load(torch::serialize::InputArchive& archive) override;

protected:
    std::unique_ptr<torch::optim::OptimizerParamState> make_param_state() ;
};

