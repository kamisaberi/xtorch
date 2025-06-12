#ifndef LAMB_OPTIMIZER_HPP
#define LAMB_OPTIMIZER_HPP

#include <torch/torch.h>
#include <torch/serialize/archive.h>

#include <cmath>
#include <vector>
#include <memory>
#include <string>
#include <cstdint>

// --- Options for LAMB Optimizer ---
struct LambOptions : torch::optim::OptimizerOptions {
    explicit LambOptions(double learning_rate = 1e-3)
        : torch::optim::OptimizerOptions() {
        this->lr(learning_rate);
    }

    // Standard Adam parameters
    TORCH_ARG(double, beta1) = 0.9;
    TORCH_ARG(double, beta2) = 0.999;
    TORCH_ARG(double, eps) = 1e-6; // LAMB often uses a slightly larger eps
    TORCH_ARG(double, weight_decay) = 0.01; // LAMB often has a default weight decay
    TORCH_ARG(double ,  lr) = 1e-6;   // For momentum on the gradient "force"
    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<torch::optim::OptimizerOptions> clone() const override;
};

// --- Parameter State for LAMB Optimizer ---
struct LambParamState : torch::optim::OptimizerParamState {
    TORCH_ARG(torch::Tensor, step);
    TORCH_ARG(torch::Tensor, exp_avg);      // m_t (EMA of gradients)
    TORCH_ARG(torch::Tensor, exp_avg_sq);   // v_t (EMA of squared gradients)

    LambParamState() = default;
    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<OptimizerParamState> clone() const override;
};

// --- LAMB Optimizer Class ---
class LAMB : public torch::optim::Optimizer {
public:
    LAMB(std::vector<torch::Tensor> params, LambOptions options);
    explicit LAMB(std::vector<torch::Tensor> params, double lr = 1e-3);

    using LossClosure = std::function<torch::Tensor()>;
    torch::Tensor step(LossClosure closure = nullptr) override;
    void save(torch::serialize::OutputArchive& archive) const override;
    void load(torch::serialize::InputArchive& archive) override;

protected:
    std::unique_ptr<torch::optim::OptimizerParamState> make_param_state() ;
};

#endif // LAMB_OPTIMIZER_HPP