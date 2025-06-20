#pragma once


#include "common.h"
// --- Options for FASFA (Factor-Wise and Statistical Factor-Wise) Optimizer ---
struct FASFAOptions :public torch::optim::OptimizerOptions
{
public:
    double lr;

    explicit FASFAOptions(double learning_rate = 1.0) // Grafting makes lr=1.0 a good default
        : torch::optim::OptimizerOptions()
    {
        this->lr = learning_rate;
    }

    // Two timescales for the "Statistical" (slow) and "Factor-Wise" (fast) components
    TORCH_ARG(double, beta_fast) = 0.9; // Decay for the fast-adapting Factor-Wise statistics
    TORCH_ARG(double, beta_slow) = 0.999; // Decay for the stable Statistical Factor-Wise statistics

    // Adam parameters for fallback and optional grafting
    TORCH_ARG(double, beta1_adam) = 0.9;
    TORCH_ARG(double, beta2_adam) = 0.999;
    TORCH_ARG(double, eps) = 1e-8; // Epsilon for Adam fallback

    TORCH_ARG(double, weight_decay) = 1e-4;
    TORCH_ARG(int, root_order) = 2; // p-th root, 2=sqrt is standard
    TORCH_ARG(long, precondition_frequency) = 100;
    TORCH_ARG(long, start_preconditioning_step) = 250;
    TORCH_ARG(std::string, grafting_type) = "SGD"; // "SGD" or "ADAM"

    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive);
    std::unique_ptr<torch::optim::OptimizerOptions> clone() const override;
};

// --- Parameter State for FASFA ---
struct FASFAParamState : torch::optim::OptimizerParamState
{
    TORCH_ARG(torch::Tensor, step);
public:
    // "Factor-Wise" (fast) and "Statistical" (slow) EMAs
    std::vector<torch::Tensor> fast_ema_factors;
    std::vector<torch::Tensor> slow_ema_factors;
    std::vector<torch::Tensor> inv_root_factors;

    // For Adam grafting and fallback
    TORCH_ARG(torch::Tensor, exp_avg); // m_t
    TORCH_ARG(torch::Tensor, exp_avg_sq); // v_t

    // FASFAParamState() = default;
    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive);
    std::unique_ptr<OptimizerParamState> clone() const override;
};

// --- FASFA Optimizer Class ---
class FASFA : public torch::optim::Optimizer
{
public:
    FASFA(std::vector<torch::Tensor> params, FASFAOptions options);

    using LossClosure = std::function<torch::Tensor()>;
    torch::Tensor step(LossClosure closure = nullptr) override;
    void save(torch::serialize::OutputArchive& archive) const override;
    void load(torch::serialize::InputArchive& archive) override;

protected:
    std::unique_ptr<torch::optim::OptimizerParamState> make_param_state();

private:
    void _fallback_to_adam(
        torch::Tensor& param,
        const torch::Tensor& grad,
        FASFAParamState& state,
        const FASFAOptions& options);

    torch::Tensor _compute_grafted_norm(
        const torch::Tensor& grad,
        FASFAParamState& state,
        const FASFAOptions& options);

    torch::Tensor _compute_matrix_inverse_root(
        const torch::Tensor& matrix,
        double damping,
        int root_order);
};

