#ifndef AMSBOUND_OPTIMIZER_HPP
#define AMSBOUND_OPTIMIZER_HPP

#include <torch/torch.h>
#include <torch/serialize/archive.h>

#include <cmath>
#include <vector>
#include <memory>
#include <string>
#include <cstdint>

// --- Options for AMSBound Optimizer ---
struct AMSBoundOptions : torch::optim::OptimizerOptions {
    explicit AMSBoundOptions(double learning_rate = 1e-3)
        : torch::optim::OptimizerOptions() {
        this->lr(learning_rate);
    }

    TORCH_ARG(double, beta1) = 0.9;
    TORCH_ARG(double, beta2) = 0.999;

    // Bounds for the adaptive learning rate
    TORCH_ARG(double, final_lr) = 0.1; // The final LR the bounds converge to
    TORCH_ARG(double, gamma) = 1e-3; // Convergence speed of the bounds

    TORCH_ARG(double, eps) = 1e-8;
    TORCH_ARG(double, weight_decay) = 0.0;
    // AMSGrad is inherent, so no toggle is needed

    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) override;
    std::unique_ptr<torch::optim::OptimizerOptions> clone() const override;
};

// --- Parameter State for AMSBound Optimizer ---
struct AMSBoundParamState : torch::optim::OptimizerParamState {
    TORCH_ARG(torch::Tensor, step);
    TORCH_ARG(torch::Tensor, exp_avg);      // m_t
    TORCH_ARG(torch::Tensor, exp_avg_sq);   // v_t
    TORCH_ARG(torch::Tensor, max_exp_avg_sq); // For AMSGrad

    AMSBoundParamState() = default;
    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) override;
    std::unique_ptr<OptimizerParamState> clone() const override;
};

// --- AMSBound Optimizer Class ---
class AMSBound : public torch::optim::Optimizer {
public:
    AMSBound(std::vector<torch::Tensor> params, AMSBoundOptions options);
    explicit AMSBound(std::vector<torch::Tensor> params, double lr = 1e-3);

    using LossClosure = std::function<torch::Tensor()>;
    torch::Tensor step(LossClosure closure = nullptr) override;
    void save(torch::serialize::OutputArchive& archive) const override;
    void load(torch::serialize::InputArchive& archive) override;

protected:
    std::unique_ptr<torch::optim::OptimizerParamState> make_param_state() override;
};

#endif // AMSBOUND_OPTIMIZER_HPP