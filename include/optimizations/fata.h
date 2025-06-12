#ifndef FATA_OPTIMIZER_HPP
#define FATA_OPTIMIZER_HPP

#include <torch/torch.h>
#include <torch/serialize/archive.h>

#include <cmath>
#include <vector>
#include <memory>
#include <string>
#include <cstdint>

// --- Options for FATA Optimizer ---
struct FATAOptions : torch::optim::OptimizerOptions {
public:
    double lr;
    explicit FATAOptions(double learning_rate = 1e-3)
        : torch::optim::OptimizerOptions() {
        this->lr=learning_rate;
    }

    // Adam parameters for the final update (on preconditioned gradients)
    TORCH_ARG(double, beta1) = 0.9;
    TORCH_ARG(double, beta2) = 0.999;
    TORCH_ARG(double, eps) = 1e-8;

    // Two timescales for factor statistics
    TORCH_ARG(double, beta_fast) = 0.95;
    TORCH_ARG(double, beta_slow) = 0.999;

    TORCH_ARG(double, weight_decay) = 1e-4;
    TORCH_ARG(int, root_order) = 2; // 2=sqrt
    TORCH_ARG(long, precondition_frequency) = 100;
    TORCH_ARG(long, start_preconditioning_step) = 250;

    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<torch::optim::OptimizerOptions> clone() const override;
};

// --- Parameter State for FATA ---
struct FATAParamState : torch::optim::OptimizerParamState {
    TORCH_ARG(torch::Tensor, step);
public:
    // Fast and slow EMAs for Kronecker factors
    std::vector<torch::Tensor> fast_ema_factors;
    std::vector<torch::Tensor> slow_ema_factors;
    std::vector<torch::Tensor> inv_root_factors;

    // Adam states for the preconditioned gradient
    TORCH_ARG(torch::Tensor, exp_avg);      // m_t
    TORCH_ARG(torch::Tensor, exp_avg_sq);   // v_t

    // FATAParamState() = default;
    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<OptimizerParamState> clone() const override;
};

// --- FATA Optimizer Class ---
class FATA : public torch::optim::Optimizer {
public:
    FATA(std::vector<torch::Tensor> params, FATAOptions options);

    using LossClosure = std::function<torch::Tensor()>;
    torch::Tensor step(LossClosure closure = nullptr) override;
    void save(torch::serialize::OutputArchive& archive) const override;
    void load(torch::serialize::InputArchive& archive) override;

protected:
    std::unique_ptr<torch::optim::OptimizerParamState> make_param_state() ;

private:
    // Fallback for 1D params remains a standard Adam update
    void _fallback_to_adam(
        torch::Tensor& param,
        const torch::Tensor& grad,
        FATAParamState& state,
        const FATAOptions& options);

    torch::Tensor _compute_matrix_inverse_root(
        const torch::Tensor& matrix,
        double damping,
        int root_order);
};

#endif // FATA_OPTIMIZER_HPP