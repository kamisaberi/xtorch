#ifndef QHM_OPTIMIZER_HPP
#define QHM_OPTIMIZER_HPP

#include <torch/torch.h>
#include <torch/serialize/archive.h>

#include <cmath>
#include <vector>
#include <memory>
#include <string>
#include <cstdint>

// --- Options for QHM Optimizer ---
struct QHMOptions : torch::optim::OptimizerOptions {
    explicit QHMOptions(double learning_rate = 0.1) // QHM often uses SGD-like LRs
        : torch::optim::OptimizerOptions() {
        this->lr(learning_rate);
    }

    // beta is the EMA decay rate for the momentum
    TORCH_ARG(double, beta) = 0.9;
    // nu is the immediate discount factor for combining current grad and momentum
    TORCH_ARG(double, nu) = 0.7;

    TORCH_ARG(double, weight_decay) = 1e-4;
    TORCH_ARG(double, lr) = 1e-6;

    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<torch::optim::OptimizerOptions> clone() const override;
};

// --- Parameter State for QHM Optimizer ---
struct QHMParamState : torch::optim::OptimizerParamState {
    TORCH_ARG(torch::Tensor, momentum_buffer); // The EMA of gradients (m_t)

    // QHMParamState() = default;
    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<OptimizerParamState> clone() const override;
};

// --- QHM Optimizer Class ---
class QHM : public torch::optim::Optimizer {
public:
    QHM(std::vector<torch::Tensor> params, QHMOptions options);
    explicit QHM(std::vector<torch::Tensor> params, double lr = 0.1);

    using LossClosure = std::function<torch::Tensor()>;
    torch::Tensor step(LossClosure closure = nullptr) override;
    void save(torch::serialize::OutputArchive& archive) const override;
    void load(torch::serialize::InputArchive& archive) override;

protected:
    std::unique_ptr<torch::optim::OptimizerParamState> make_param_state() ;
};

#endif // QHM_OPTIMIZER_HPP