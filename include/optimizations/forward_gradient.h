#pragma once


#include "common.h"
// --- Options for ForwardGradient (Lookahead) Optimizer ---
struct ForwardGradientOptions : torch::optim::OptimizerOptions
{
public:
    double lr;

    explicit ForwardGradientOptions(double learning_rate = 1e-3)
        : torch::optim::OptimizerOptions()
    {
        this->lr = learning_rate; // LR for the inner Adam optimizer
    }

    // Lookahead parameters
    TORCH_ARG(double, alpha) = 0.5; // Interpolation factor for slow weights
    TORCH_ARG(long, k) = 6; // Sync/update frequency for slow weights

    // Inner Adam optimizer parameters
    TORCH_ARG(double, beta1) = 0.9;
    TORCH_ARG(double, beta2) = 0.999;
    TORCH_ARG(double, eps) = 1e-8;
    TORCH_ARG(double, weight_decay) = 0.0;

    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive);
    std::unique_ptr<torch::optim::OptimizerOptions> clone() const override;
};

// --- Parameter State for ForwardGradient (Lookahead) ---
struct ForwardGradientParamState : torch::optim::OptimizerParamState
{
    TORCH_ARG(torch::Tensor, step);

    // Slow weights (the main parameters for lookahead)
    TORCH_ARG(torch::Tensor, slow_param);

    // Inner Adam optimizer states
    TORCH_ARG(torch::Tensor, exp_avg); // m_t
    TORCH_ARG(torch::Tensor, exp_avg_sq); // v_t
public:
    // ForwardGradientParamState() = default;
    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive);
    std::unique_ptr<OptimizerParamState> clone() const override;
};

// --- ForwardGradient (Lookahead) Optimizer Class ---
class ForwardGradient : public torch::optim::Optimizer
{
public:
    ForwardGradient(std::vector<torch::Tensor> params, ForwardGradientOptions options);

    using LossClosure = std::function<torch::Tensor()>;
    torch::Tensor step(LossClosure closure = nullptr) override;
    void save(torch::serialize::OutputArchive& archive) const override;
    void load(torch::serialize::InputArchive& archive) override;

protected:
    std::unique_ptr<torch::optim::OptimizerParamState> make_param_state();
};

