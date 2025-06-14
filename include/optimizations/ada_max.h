#ifndef ADAMAX_OPTIMIZER_HPP
#define ADAMAX_OPTIMIZER_HPP

#include <torch/torch.h>
#include <torch/serialize/archive.h>

#include <cmath>
#include <vector>
#include <memory>
#include <string>
#include <cstdint>

// --- Options for AdaMax Optimizer ---
struct AdaMaxOptions : torch::optim::OptimizerOptions {
    explicit AdaMaxOptions(double learning_rate = 2e-3) // Adamax often uses this default LR
        : torch::optim::OptimizerOptions() {
        this->lr(learning_rate);
    }

    TORCH_ARG(double, beta1) = 0.9;
    TORCH_ARG(double, beta2) = 0.999;
    TORCH_ARG(double, eps) = 1e-8;
    TORCH_ARG(double, weight_decay) = 0.0;

    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) override;
    std::unique_ptr<torch::optim::OptimizerOptions> clone() const override;
};

// --- Parameter State for AdaMax Optimizer ---
struct AdaMaxParamState : torch::optim::OptimizerParamState {
    TORCH_ARG(torch::Tensor, step);
    TORCH_ARG(torch::Tensor, exp_avg);      // m_t (EMA of gradients)
    TORCH_ARG(torch::Tensor, exp_inf_norm); // u_t in the paper (EMA of infinity norm)

    AdaMaxParamState() = default;
    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) override;
    std::unique_ptr<OptimizerParamState> clone() const override;
};

// --- AdaMax Optimizer Class ---
class AdaMax : public torch::optim::Optimizer {
public:
    AdaMax(std::vector<torch::Tensor> params, AdaMaxOptions options);
    explicit AdaMax(std::vector<torch::Tensor> params, double lr = 2e-3);

    using LossClosure = std::function<torch::Tensor()>;
    torch::Tensor step(LossClosure closure = nullptr) override;
    void save(torch::serialize::OutputArchive& archive) const override;
    void load(torch::serialize::InputArchive& archive) override;

protected:
    std::unique_ptr<torch::optim::OptimizerParamState> make_param_state() override;
};

#endif // ADAMAX_OPTIMIZER_HPP