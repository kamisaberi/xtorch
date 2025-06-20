#pragma once


#include "common.h"
// --- Options for SM3 Optimizer ---
struct SM3Options : torch::optim::OptimizerOptions {
    explicit SM3Options(double learning_rate = 0.1) // SM3 often uses SGD-like LR
        : torch::optim::OptimizerOptions() {
        this->lr(learning_rate);
    }

    // Momentum parameter
    TORCH_ARG(double, beta) = 0.9;
    TORCH_ARG(double, eps) = 1e-8;
    TORCH_ARG(double, lr) = 1e-6;

    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<torch::optim::OptimizerOptions> clone() const override;
};

// --- Parameter State for SM3 Optimizer ---
struct SM3ParamState : torch::optim::OptimizerParamState {
    // For 2D+ tensors
    torch::Tensor row_accumulator;
    torch::Tensor col_accumulator;

    // For all tensors
    TORCH_ARG(torch::Tensor, momentum_buffer);

    // SM3ParamState() = default;
    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<OptimizerParamState> clone() const override;
};

// --- SM3 Optimizer Class ---
class SM3 : public torch::optim::Optimizer {
public:
    SM3(std::vector<torch::Tensor> params, SM3Options options);
    explicit SM3(std::vector<torch::Tensor> params, double lr = 0.1);

    using LossClosure = std::function<torch::Tensor()>;
    torch::Tensor step(LossClosure closure = nullptr) override;
    void save(torch::serialize::OutputArchive& archive) const override;
    void load(torch::serialize::InputArchive& archive) override;

protected:
    std::unique_ptr<torch::optim::OptimizerParamState> make_param_state() ;
};

