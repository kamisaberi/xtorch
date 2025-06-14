#ifndef ADASHIFT_OPTIMIZER_HPP
#define ADASHIFT_OPTIMIZER_HPP

#include <torch/torch.h>
#include <torch/serialize/archive.h>

#include <cmath>
#include <vector>
#include <memory>
#include <string>
#include <cstdint>
#include <deque>

// --- Options for AdaShift Optimizer ---
struct AdaShiftOptions :public torch::optim::OptimizerOptions {
    explicit AdaShiftOptions(double learning_rate = 1e-3)
        : torch::optim::OptimizerOptions() {
        this->lr(learning_rate);
    }

    TORCH_ARG(double, beta1) = 0.9;
    TORCH_ARG(double, beta2) = 0.999;
    TORCH_ARG(double, eps) = 1e-8;
    TORCH_ARG(double, weight_decay) = 0.0;

    // AdaShift specific parameter: the lookback delay
    TORCH_ARG(long, lookback_d) = 10;

    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) override;
    std::unique_ptr<torch::optim::OptimizerOptions> clone() const override;
};

// --- Parameter State for AdaShift Optimizer ---
struct AdaShiftParamState :public torch::optim::OptimizerParamState {
    TORCH_ARG(torch::Tensor, step);
    TORCH_ARG(torch::Tensor, exp_avg);      // m_t (current momentum)
    TORCH_ARG(torch::Tensor, exp_avg_sq);   // v_t (historical variance)
public:
    // A queue to store recent squared gradients for the delayed update
    std::deque<torch::Tensor> grad_sq_history;

    AdaShiftParamState() = default;
    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) override;
    std::unique_ptr<OptimizerParamState> clone() const override;
};

// --- AdaShift Optimizer Class ---
class AdaShift : public torch::optim::Optimizer {
public:
    AdaShift(std::vector<torch::Tensor> params, AdaShiftOptions options);
    explicit AdaShift(std::vector<torch::Tensor> params, double lr = 1e-3);

    using LossClosure = std::function<torch::Tensor()>;
    torch::Tensor step(LossClosure closure = nullptr) override;
    void save(torch::serialize::OutputArchive& archive) const override;
    void load(torch::serialize::InputArchive& archive) override;

protected:
    std::unique_ptr<torch::optim::OptimizerParamState> make_param_state() override;
};

#endif // ADASHIFT_OPTIMIZER_HPP