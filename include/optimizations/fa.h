#pragma once


#include "common.h"

// --- Options for FA Optimizer ---
struct FAOptions :public torch::optim::OptimizerOptions {
    double lr;
    explicit FAOptions(double learning_rate = 1e-3)
        : torch::optim::OptimizerOptions() {
        this->lr=learning_rate;
    }

    // Adam-like momentum parameters
    TORCH_ARG(double, beta1) = 0.9;
    // EMA decay for factor statistics
    TORCH_ARG(double, beta2) = 0.999;

    TORCH_ARG(double, eps) = 1e-8; // Epsilon for Adam fallback
    TORCH_ARG(double, damping) = 1e-6; // Damping for matrix inverse root
    TORCH_ARG(double, weight_decay) = 1e-4;
    TORCH_ARG(int, root_order) = 2; // p-th root for factors, 2=sqrt
    TORCH_ARG(long, update_frequency) = 100; // How often to compute inverse roots

    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<torch::optim::OptimizerOptions> clone() const override;
};

// --- Parameter State for FA Optimizer ---
struct FAParamState :public torch::optim::OptimizerParamState {
    TORCH_ARG(torch::Tensor, step);
    TORCH_ARG(torch::Tensor, momentum); // m_t (momentum on preconditioned grad)
public:
    // Statistics for P (row factor) and Q (column factor)
    torch::Tensor p_ema;
    torch::Tensor q_ema;

    // Preconditioner factors
    torch::Tensor p_inv_root;
    torch::Tensor q_inv_root;

    // For Adam fallback
    TORCH_ARG(torch::Tensor, exp_avg_sq);

    // FAParamState() = default;
    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<OptimizerParamState> clone() const override;
};

// --- FA Optimizer Class ---
class FAOptimizer : public torch::optim::Optimizer {
public:
    FAOptimizer(std::vector<torch::Tensor> params, FAOptions options);

    using LossClosure = std::function<torch::Tensor()>;
    torch::Tensor step(LossClosure closure = nullptr) override;
    void save(torch::serialize::OutputArchive& archive) const override;
    void load(torch::serialize::InputArchive& archive) override;

protected:
    std::unique_ptr<torch::optim::OptimizerParamState> make_param_state() ;

private:
    void _fallback_to_adam(
        torch::Tensor& param,
        const torch::Tensor& grad,
        FAParamState& state,
        const FAOptions& options);

    torch::Tensor _compute_matrix_inverse_root(
        const torch::Tensor& matrix,
        double damping,
        int root_order);
};

