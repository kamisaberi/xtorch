#ifndef NADAM_OPTIMIZER_HPP
#define NADAM_OPTIMIZER_HPP

#include <torch/torch.h>
#include <torch/serialize/archive.h>

#include <cmath>
#include <vector>
#include <memory>
#include <string>
#include <cstdint>

// --- Options for NAdam Optimizer ---
struct NAdamOptions : torch::optim::OptimizerOptions {
    explicit NAdamOptions(double learning_rate = 2e-3) // NAdam often uses a slightly different default LR
        : torch::optim::OptimizerOptions() {
        this->lr(learning_rate);
    }

    // Adam-like parameters
    TORCH_ARG(double, beta1) = 0.9;
    TORCH_ARG(double, beta2) = 0.999;
    TORCH_ARG(double, eps) = 1e-8;
    TORCH_ARG(double, weight_decay) = 0.0;
    TORCH_ARG(double, lr) = 1e-6;

    // NAdam specific: momentum decay schedule (optional, we use fixed beta1)
    // TORCH_ARG(double, momentum_decay) = 4e-3;

    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<torch::optim::OptimizerOptions> clone() const override;
};

// --- Parameter State for NAdam Optimizer ---
struct NAdamParamState : torch::optim::OptimizerParamState {
    TORCH_ARG(torch::Tensor, step);
    TORCH_ARG(torch::Tensor, exp_avg);      // m_t (EMA of gradients)
    TORCH_ARG(torch::Tensor, exp_avg_sq);   // v_t (EMA of squared gradients)

    NAdamParamState() = default;
    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<OptimizerParamState> clone() const override;
};

// --- NAdam Optimizer Class ---
class NAdam : public torch::optim::Optimizer {
public:
    NAdam(std::vector<torch::Tensor> params, NAdamOptions options);
    explicit NAdam(std::vector<torch::Tensor> params, double lr = 2e-3);

    using LossClosure = std::function<torch::Tensor()>;
    torch::Tensor step(LossClosure closure = nullptr) override;
    void save(torch::serialize::OutputArchive& archive) const override;
    void load(torch::serialize::InputArchive& archive) override;

protected:
    std::unique_ptr<torch::optim::OptimizerParamState> make_param_state() ;
};

#endif // NADAM_OPTIMIZER_HPP