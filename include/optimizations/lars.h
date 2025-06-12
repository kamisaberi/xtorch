#ifndef LARS_OPTIMIZER_HPP
#define LARS_OPTIMIZER_HPP

#include <torch/torch.h>
#include <torch/serialize/archive.h>

#include <cmath>
#include <vector>
#include <memory>
#include <string>
#include <cstdint>

// --- Options for LARS Optimizer ---
struct LARSOptions : torch::optim::OptimizerOptions {
    explicit LARSOptions(double learning_rate = 0.1) // Base LR can be higher for LARS
        : torch::optim::OptimizerOptions() {
        this->lr(learning_rate);
    }

    // SGD with Momentum parameters
    TORCH_ARG(double, momentum) = 0.9;
    TORCH_ARG(double, weight_decay) = 1e-4;
    TORCH_ARG(double ,  lr) = 1e-6;   // For momentum on the gradient "force"
    // LARS specific parameters
    TORCH_ARG(double, trust_coefficient) = 0.001; // The 'eta' in the LARS paper
    TORCH_ARG(double, eps) = 1e-8; // Epsilon to prevent division by zero in norm calculations

    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<torch::optim::OptimizerOptions> clone() const override;
};

// --- Parameter State for LARS Optimizer ---
struct LARSParamState : torch::optim::OptimizerParamState {
    TORCH_ARG(torch::Tensor, momentum_buffer);

    LARSParamState() = default;
    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<OptimizerParamState> clone() const override;
};

// --- LARS Optimizer Class ---
class LARS : public torch::optim::Optimizer {
public:
    LARS(std::vector<torch::Tensor> params, LARSOptions options);
    explicit LARS(std::vector<torch::Tensor> params, double lr = 0.1);

    using LossClosure = std::function<torch::Tensor()>;
    torch::Tensor step(LossClosure closure = nullptr) override;
    void save(torch::serialize::OutputArchive& archive) const override;
    void load(torch::serialize::InputArchive& archive) override;

protected:
    std::unique_ptr<torch::optim::OptimizerParamState> make_param_state() ;
};

#endif // LARS_OPTIMIZER_HPP