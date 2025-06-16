#ifndef MAS_OPTIMIZER_HPP
#define MAS_OPTIMIZER_HPP

#include <torch/torch.h>
#include <torch/serialize/archive.h>

#include <cmath>
#include <vector>
#include <memory>
#include <string>
#include <cstdint>

#include "../data_loaders/extended_data_loader.h"

// --- Options for MAS Optimizer ---
struct MASOptions : torch::optim::OptimizerOptions {
    explicit MASOptions(double learning_rate = 1e-3)
        : torch::optim::OptimizerOptions() {
        this->lr(learning_rate);
    }

    // Adam parameters for the inner optimizer
    TORCH_ARG(double, beta1) = 0.9;
    TORCH_ARG(double, beta2) = 0.999;
    TORCH_ARG(double, eps) = 1e-8;
    TORCH_ARG(double, lr) = 1e-6;

    // MAS specific parameter
    TORCH_ARG(double, lambda) = 1.0; // Regularization strength

    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<torch::optim::OptimizerOptions> clone() const override;
};

// --- Parameter State for MAS ---
struct MASParamState : torch::optim::OptimizerParamState {
    TORCH_ARG(torch::Tensor, step);

    // MAS specific state
    TORCH_ARG(torch::Tensor, importance);        // Omega
    TORCH_ARG(torch::Tensor, optimal_param);    // theta*

    // Inner Adam states
    TORCH_ARG(torch::Tensor, exp_avg);
    TORCH_ARG(torch::Tensor, exp_avg_sq);

    // MASParamState() = default;
    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<OptimizerParamState> clone() const override;
};

// --- MAS Optimizer Class ---
class MAS : public torch::optim::Optimizer {
public:
    MAS(std::vector<torch::Tensor> params, MASOptions options);

    using LossClosure = std::function<torch::Tensor()>;
    torch::Tensor step(LossClosure closure = nullptr) override;

    // Special method to be called after a task is finished
    void compute_and_update_importance(xt::Module& model, xt::dataloaders::ExtendedDataLoader& data_loader);

    void save(torch::serialize::OutputArchive& archive) const override;
    void load(torch::serialize::InputArchive& archive) override;

protected:
    std::unique_ptr<torch::optim::OptimizerParamState> make_param_state() ;
};

#endif // MAS_OPTIMIZER_HPP