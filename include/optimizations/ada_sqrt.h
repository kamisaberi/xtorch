#ifndef ADASQRT_OPTIMIZER_HPP
#define ADASQRT_OPTIMIZER_HPP

#include <torch/torch.h>
#include <torch/serialize/archive.h>

#include <cmath>
#include <vector>
#include <memory>
#include <string>
#include <cstdint>

// --- Options for AdaSqrt Optimizer ---
struct AdaSqrtOptions : torch::optim::OptimizerOptions {
    explicit AdaSqrtOptions(double learning_rate = 1e-3)
        : torch::optim::OptimizerOptions() {
        this->lr(learning_rate);
    }

    TORCH_ARG(double, beta1) = 0.9;   // For momentum (m_t)
    TORCH_ARG(double, beta2) = 0.999; // For the adaptive term (v_t)
    TORCH_ARG(double, eps) = 1e-8;
    TORCH_ARG(double, weight_decay) = 0.0;
    TORCH_ARG(double ,  lr) = 1e-6;

    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<torch::optim::OptimizerOptions> clone() const override;
};

// --- Parameter State for AdaSqrt Optimizer ---
struct AdaSqrtParamState : torch::optim::OptimizerParamState {
    TORCH_ARG(torch::Tensor, step);
    TORCH_ARG(torch::Tensor, exp_avg);      // m_t (EMA of gradients)
    TORCH_ARG(torch::Tensor, exp_avg_abs);  // v_t (EMA of absolute gradients)

    // AdaSqrtParamState() = default;
    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<OptimizerParamState> clone() const override;
};

// --- AdaSqrt Optimizer Class ---
class AdaSqrt : public torch::optim::Optimizer {
public:
    AdaSqrt(std::vector<torch::Tensor> params, AdaSqrtOptions options);
    explicit AdaSqrt(std::vector<torch::Tensor> params, double lr = 1e-3);

    using LossClosure = std::function<torch::Tensor()>;
    torch::Tensor step(LossClosure closure = nullptr) override;
    void save(torch::serialize::OutputArchive& archive) const override;
    void load(torch::serialize::InputArchive& archive) override;

protected:
    std::unique_ptr<torch::optim::OptimizerParamState> make_param_state() ;
};

#endif // ADASQRT_OPTIMIZER_HPP