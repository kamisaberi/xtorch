#ifndef AMSGRAD_OPTIMIZER_HPP
#define AMSGRAD_OPTIMIZER_HPP

#include <torch/torch.h>
#include <torch/serialize/archive.h>

#include <cmath>
#include <vector>
#include <memory>
#include <string>
#include <cstdint>

// --- Options for AMSGrad Optimizer ---
struct AMSGradOptions : torch::optim::OptimizerOptions {
    explicit AMSGradOptions(double learning_rate = 1e-3)
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

// --- Parameter State for AMSGrad Optimizer ---
struct AMSGradParamState : torch::optim::OptimizerParamState {
    TORCH_ARG(torch::Tensor, step);
    TORCH_ARG(torch::Tensor, exp_avg);      // m_t
    TORCH_ARG(torch::Tensor, exp_avg_sq);   // v_t
    TORCH_ARG(torch::Tensor, max_exp_avg_sq); // The key AMSGrad state

    AMSGradParamState() = default;
    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) override;
    std::unique_ptr<OptimizerParamState> clone() const override;
};

// --- AMSGrad Optimizer Class ---
class AMSGrad : public torch::optim::Optimizer {
public:
    AMSGrad(std::vector<torch::Tensor> params, AMSGradOptions options);
    explicit AMSGrad(std::vector<torch::Tensor> params, double lr = 1e-3);

    using LossClosure = std::function<torch::Tensor()>;
    torch::Tensor step(LossClosure closure = nullptr) override;
    void save(torch::serialize::OutputArchive& archive) const override;
    void load(torch::serialize::InputArchive& archive) override;

protected:
    std::unique_ptr<torch::optim::OptimizerParamState> make_param_state() override;
};

#endif // AMSGRAD_OPTIMIZER_HPP