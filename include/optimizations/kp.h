#pragma once


#include "common.h"
// --- Options for KP Optimizer ---
struct KPOptions : torch::optim::OptimizerOptions {
    explicit KPOptions(double learning_rate = 1.0) // Projection provides step size, so lr can be 1.0
        : torch::optim::OptimizerOptions() {
        this->lr(learning_rate);
    }

    // Momentum on the final projected update
    TORCH_ARG(double, beta1) = 0.9;
    // EMA decay for Kronecker factor statistics
    TORCH_ARG(double, beta2) = 0.999;
    TORCH_ARG(double ,  lr) = 1e-6;   // For momentum on the gradient "force"
    TORCH_ARG(double, damping) = 1e-6; // Damping for matrix inverse root
    TORCH_ARG(double, weight_decay) = 1e-4;
    TORCH_ARG(int, root_order) = 2; // p-th root, 2=sqrt
    TORCH_ARG(long, precondition_frequency) = 100;

    // Adam parameters for the 1D fallback
    TORCH_ARG(double, fallback_beta1) = 0.9;
    TORCH_ARG(double, fallback_beta2) = 0.999;
    TORCH_ARG(double, fallback_eps) = 1e-8;

    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<torch::optim::OptimizerOptions> clone() const override;
};

// --- Parameter State for KP Optimizer ---
struct KPParamState : torch::optim::OptimizerParamState {
    TORCH_ARG(torch::Tensor, step);
    TORCH_ARG(torch::Tensor, momentum_buffer); // Momentum on the final update
public:
    // Kronecker factor statistics and inverse roots
    torch::Tensor l_ema; // Left factor
    torch::Tensor r_ema; // Right factor
    torch::Tensor l_inv_root;
    torch::Tensor r_inv_root;

    // For Adam fallback
    TORCH_ARG(torch::Tensor, fallback_exp_avg);
    TORCH_ARG(torch::Tensor, fallback_exp_avg_sq);

    // KPParamState() = default;
    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<OptimizerParamState> clone() const override;
};

// --- KP Optimizer Class ---
class KPOptimizer : public torch::optim::Optimizer {
public:
    KPOptimizer(std::vector<torch::Tensor> params, KPOptions options);

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
        KPParamState& state,
        const KPOptions& options);

    torch::Tensor _compute_matrix_inverse_root(
        const torch::Tensor& matrix,
        double damping,
        int root_order);
};

