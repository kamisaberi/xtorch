#ifndef QHADAM_OPTIMIZER_HPP
#define QHADAM_OPTIMIZER_HPP

#include <torch/torch.h>
#include <torch/serialize/archive.h>

#include <cmath>
#include <vector>
#include <memory>
#include <string>
#include <cstdint>

// --- Options for QHAdam Optimizer ---
struct QHAdamOptions : torch::optim::OptimizerOptions {
    explicit QHAdamOptions(double learning_rate = 1e-3)
        : torch::optim::OptimizerOptions() {
        this->lr(learning_rate);
    }

    // nu1 is the immediate discount factor for the gradient (1-nu1) and momentum (nu1)
    TORCH_ARG(double, nu1) = 0.7;
    // nu2 is the immediate discount factor for the squared gradient
    TORCH_ARG(double, nu2) = 1.0; // nu2=1.0 makes it RMSProp-like, which is common

    // beta1 and beta2 are the EMA decay rates, same as Adam
    TORCH_ARG(double, beta1) = 0.9;
    TORCH_ARG(double, beta2) = 0.999;

    TORCH_ARG(double, weight_decay) = 0.0;
    TORCH_ARG(double, eps) = 1e-8;
    TORCH_ARG(double, lr) = 1e-6;

    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<torch::optim::OptimizerOptions> clone() const override;
};

// --- Parameter State for QHAdam ---
struct QHAdamParamState : torch::optim::OptimizerParamState {
    TORCH_ARG(torch::Tensor, exp_avg);      // m_t (EMA of gradients)
    TORCH_ARG(torch::Tensor, exp_avg_sq);   // v_t (EMA of squared gradients)

    // QHAdamParamState() = default;
    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<OptimizerParamState> clone() const override;
};

// --- QHAdam Optimizer Class ---
class QHAdam : public torch::optim::Optimizer {
public:
    QHAdam(std::vector<torch::Tensor> params, QHAdamOptions options);
    explicit QHAdam(std::vector<torch::Tensor> params, double lr = 1e-3);

    using LossClosure = std::function<torch::Tensor()>;
    torch::Tensor step(LossClosure closure = nullptr) override;
    void save(torch::serialize::OutputArchive& archive) const override;
    void load(torch::serialize::InputArchive& archive) override;

protected:
    std::unique_ptr<torch::optim::OptimizerParamState> make_param_state() ;
};

#endif // QHADAM_OPTIMIZER_HPP