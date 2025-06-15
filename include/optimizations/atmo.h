#ifndef ATMO_OPTIMIZER_HPP
#define ATMO_OPTIMIZER_HPP

#include <torch/torch.h>
#include <torch/serialize/archive.h>

#include <cmath>
#include <vector>
#include <memory>
#include <string>
#include <cstdint>

// --- Options for ATMO Optimizer ---
struct ATMOOptions : torch::optim::OptimizerOptions {
    explicit ATMOOptions(double learning_rate = 0.1) // Uses SGD-like LRs
        : torch::optim::OptimizerOptions() {
        this->lr(learning_rate);
        // Default betas representing different timescales
        betas_ = {0.0, 0.9, 0.99};
    }
    TORCH_ARG(double ,  lr) = 1e-6;
    // Vector of beta values for each momentum buffer
    std::vector<double> betas_;
    ATMOOptions& betas(const std::vector<double>& new_betas) {
        betas_ = new_betas;
        return *this;
    }
    const std::vector<double>& betas() const { return betas_; }

    TORCH_ARG(double, weight_decay) = 1e-4;
    TORCH_ARG(double, temperature) = 1.0; // Softmax temperature for attention weights
    TORCH_ARG(double, eps) = 1e-8; // For stable norm calculation

    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<torch::optim::OptimizerOptions> clone() const override;
};

// --- Parameter State for ATMO Optimizer ---
struct ATMOParamState : torch::optim::OptimizerParamState {
    // A vector of momentum buffers, one for each beta
    std::vector<torch::Tensor> momentum_buffers;

    ATMOParamState() = default;
    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<OptimizerParamState> clone() const override;
};

// --- ATMO Optimizer Class ---
class ATMO : public torch::optim::Optimizer {
public:
    ATMO(std::vector<torch::Tensor> params, ATMOOptions options);
    explicit ATMO(std::vector<torch::Tensor> params, double lr = 0.1);

    using LossClosure = std::function<torch::Tensor()>;
    torch::Tensor step(LossClosure closure = nullptr) override;
    void save(torch::serialize::OutputArchive& archive) const override;
    void load(torch::serialize::InputArchive& archive) override;

protected:
    std::unique_ptr<torch::optim::OptimizerParamState> make_param_state() ;
};

#endif // ATMO_OPTIMIZER_HPP