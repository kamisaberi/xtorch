#ifndef ADABOUND_OPTIMIZER_HPP
#define ADABOUND_OPTIMIZER_HPP

#include <torch/torch.h>
#include <torch/serialize/archive.h>

#include <cmath>
#include <vector>
#include <memory>
#include <string>
#include <cstdint>

// --- Options for AdaBound Optimizer ---
struct AdaBoundOptions : torch::optim::OptimizerOptions {
    explicit AdaBoundOptions(double learning_rate = 1e-3)
        : torch::optim::OptimizerOptions() {
        this->lr(learning_rate);
        final_lr(0.1); // Default final_lr, often higher than initial
    }

    TORCH_ARG(double, beta1) = 0.9;
    TORCH_ARG(double, beta2) = 0.999;

    // final_lr is the rate the bounds converge to.
    TORCH_ARG(double, final_lr);
    // gamma controls the speed of convergence of the bounds.
    TORCH_ARG(double, gamma) = 1e-3;

    TORCH_ARG(double, eps) = 1e-8;
    TORCH_ARG(double, weight_decay) = 0.0;
    TORCH_ARG(double ,  lr) = 1e-6;

    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<torch::optim::OptimizerOptions> clone() const override;
};

// --- Parameter State for AdaBound Optimizer ---
struct AdaBoundParamState : torch::optim::OptimizerParamState {
    TORCH_ARG(torch::Tensor, step);
    TORCH_ARG(torch::Tensor, exp_avg);      // m_t
    TORCH_ARG(torch::Tensor, exp_avg_sq);   // v_t

    // AdaBoundParamState() = default;
    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<OptimizerParamState> clone() const override;
};

// --- AdaBound Optimizer Class ---
class AdaBound : public torch::optim::Optimizer {
public:
    AdaBound(std::vector<torch::Tensor> params, AdaBoundOptions options);
    explicit AdaBound(std::vector<torch::Tensor> params, double lr = 1e-3);

    using LossClosure = std::function<torch::Tensor()>;
    torch::Tensor step(LossClosure closure = nullptr) override;
    void save(torch::serialize::OutputArchive& archive) const override;
    void load(torch::serialize::InputArchive& archive) override;

protected:
    std::unique_ptr<torch::optim::OptimizerParamState> make_param_state() ;
};

#endif // ADABOUND_OPTIMIZER_HPP