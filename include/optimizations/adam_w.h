#ifndef ADAMW_OPTIMIZER_HPP
#define ADAMW_OPTIMIZER_HPP

#include <torch/torch.h>
#include <torch/serialize/archive.h>

#include <cmath>
#include <vector>
#include <memory>
#include <string>
#include <cstdint>

// --- Options for AdamW Optimizer ---
// The options are identical to Adam, the difference is in the implementation.
struct AdamWOptions : torch::optim::OptimizerOptions {
    explicit AdamWOptions(double learning_rate = 1e-3)
        : torch::optim::OptimizerOptions() {
        this->lr(learning_rate);
    }

    TORCH_ARG(double, beta1) = 0.9;
    TORCH_ARG(double, beta2) = 0.999;
    TORCH_ARG(double, eps) = 1e-8;
    TORCH_ARG(double, weight_decay) = 1e-2; // A non-zero default is common for AdamW
    TORCH_ARG(bool, amsgrad) = false; // Optional AMSGrad variant
    TORCH_ARG(double ,  lr) = 1e-6;

    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<torch::optim::OptimizerOptions> clone() const override;
};

// --- Parameter State for AdamW Optimizer ---
// The state is also identical to Adam's.
struct AdamWParamState : torch::optim::OptimizerParamState {
    TORCH_ARG(torch::Tensor, step);
    TORCH_ARG(torch::Tensor, exp_avg);      // m_t
    TORCH_ARG(torch::Tensor, exp_avg_sq);   // v_t
    TORCH_ARG(torch::Tensor, max_exp_avg_sq); // For AMSGrad

    // AdamWParamState() = default;
    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<OptimizerParamState> clone() const override;
};

// --- AdamW Optimizer Class ---
class AdamW : public torch::optim::Optimizer {
public:
    AdamW(std::vector<torch::Tensor> params, AdamWOptions options);
    explicit AdamW(std::vector<torch::Tensor> params, double lr = 1e-3);

    using LossClosure = std::function<torch::Tensor()>;
    torch::Tensor step(LossClosure closure = nullptr) override;
    void save(torch::serialize::OutputArchive& archive) const override;
    void load(torch::serialize::InputArchive& archive) override;

protected:
    std::unique_ptr<torch::optim::OptimizerParamState> make_param_state() ;
};

#endif // ADAMW_OPTIMIZER_HPP