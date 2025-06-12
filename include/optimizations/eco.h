#ifndef ECO_OPTIMIZER_HPP
#define ECO_OPTIMIZER_HPP

#include <torch/torch.h>
#include <torch/serialize/archive.h>

#include <cmath>
#include <vector>
#include <memory>
#include <string>
#include <cstdint>

// --- Options for ECO ---
struct ECOOptions : public torch::optim::OptimizerOptions {
    double lr;
    explicit ECOOptions(double learning_rate = 1.0) // Grafting makes lr=1.0 a good default
        : torch::optim::OptimizerOptions() {
        this->lr=learning_rate;
    }

    // Two timescales for EMAs
    TORCH_ARG(double, beta_fast) = 0.9;
    TORCH_ARG(double, beta_slow) = 0.999;

    // Low-level Adam parameters for grafting and fallback
    TORCH_ARG(double, beta1_adam) = 0.9;
    TORCH_ARG(double, beta2_adam) = 0.999;
    TORCH_ARG(double, eps) = 1e-8; // Epsilon for Adam fallback and grafting

    TORCH_ARG(double, weight_decay) = 1e-4;
    TORCH_ARG(int, root_order) = 2; // p-th root, ECO paper uses 2 (sqrt)
    TORCH_ARG(long, precondition_frequency) = 100;
    TORCH_ARG(long, start_preconditioning_step) = 250;
    TORCH_ARG(std::string, grafting_type) = "SGD"; // "SGD" or "ADAM"

    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<torch::optim::OptimizerOptions> clone() const override;
};

// --- Parameter State for ECO ---
struct ECOParamState :public torch::optim::OptimizerParamState {
    TORCH_ARG(torch::Tensor, step);
public:
    // Two-timescale EMAs for Kronecker factors
    std::vector<torch::Tensor> fast_ema_factors; // L_fast, R_fast
    std::vector<torch::Tensor> slow_ema_factors; // L_slow, R_slow
    std::vector<torch::Tensor> inv_root_factors; // L^{-1/p}, R^{-1/p}

    // For Adam grafting and fallback
    TORCH_ARG(torch::Tensor, exp_avg);      // m_t
    TORCH_ARG(torch::Tensor, exp_avg_sq);   // v_t

    // ECOParamState() = default;
    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<OptimizerParamState> clone() const override;
};

// --- ECO Optimizer Class ---
class ECO : public torch::optim::Optimizer {
public:
    ECO(std::vector<torch::Tensor> params, ECOOptions options);

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
        ECOParamState& state,
        const ECOOptions& options);

    torch::Tensor _compute_grafted_norm(
        const torch::Tensor& grad,
        ECOParamState& state,
        const ECOOptions& options);

    torch::Tensor _compute_matrix_inverse_root(
        const torch::Tensor& matrix,
        double damping,
        int root_order);
};

#endif // ECO_OPTIMIZER_HPP