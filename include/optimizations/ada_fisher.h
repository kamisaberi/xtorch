#ifndef ADAFISHER_OPTIMIZER_HPP
#define ADAFISHER_OPTIMIZER_HPP

#include <torch/torch.h>
#include <torch/serialize/archive.h>

#include <cmath>
#include <vector>
#include <memory>
#include <string>
#include <cstdint>

// --- Options for AdaFisher Optimizer ---
struct AdaFisherOptions : torch::optim::OptimizerOptions {
    explicit AdaFisherOptions(double learning_rate = 1e-3)
        : torch::optim::OptimizerOptions() {
        this->lr(learning_rate);
    }

    // Decay rate for momentum on the natural gradient
    TORCH_ARG(double, beta1) = 0.9;
    // Decay rate for the EMA of the Fisher information estimate
    TORCH_ARG(double, beta2) = 0.999;

    TORCH_ARG(double, eps) = 1e-8; // Damping term for the denominator
    TORCH_ARG(double, weight_decay) = 0.0;
    TORCH_ARG(double ,  lr) = 1e-6;

    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<torch::optim::OptimizerOptions> clone() const override;
};

// --- Parameter State for AdaFisher Optimizer ---
struct AdaFisherParamState : torch::optim::OptimizerParamState {
    TORCH_ARG(torch::Tensor, step);
    TORCH_ARG(torch::Tensor, exp_avg);          // m_t (Momentum on the natural gradient)
    TORCH_ARG(torch::Tensor, fisher_diag_ema);  // F_t (EMA of squared gradients)

    // AdaFisherParamState() = default;
    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<OptimizerParamState> clone() const override;
};

// --- AdaFisher Optimizer Class ---
class AdaFisher : public torch::optim::Optimizer {
public:
    AdaFisher(std::vector<torch::Tensor> params, AdaFisherOptions options);
    explicit AdaFisher(std::vector<torch::Tensor> params, double lr = 1e-3);

    using LossClosure = std::function<torch::Tensor()>;
    torch::Tensor step(LossClosure closure = nullptr) override;
    void save(torch::serialize::OutputArchive& archive) const override;
    void load(torch::serialize::InputArchive& archive) override;

protected:
    std::unique_ptr<torch::optim::OptimizerParamState> make_param_state() ;
};

#endif // ADAFISHER_OPTIMIZER_HPP