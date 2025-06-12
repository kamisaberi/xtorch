#ifndef GRADIENT_CHECKPOINTING_OPTIMIZER_HPP
#define GRADIENT_CHECKPOINTING_OPTIMIZER_HPP

#include <torch/torch.h>
#include <torch/serialize/archive.h>
#include <torch/utils.h> // For torch::utils::checkpoint

#include <cmath>
#include <vector>
#include <memory>
#include <string>
#include <cstdint>
#include <functional>

// --- Options for the inner Adam optimizer ---
struct GCO_InnerOptions : torch::optim::OptimizerOptions
{
    double lr;

    explicit GCO_InnerOptions(double learning_rate = 1e-3)
        : torch::optim::OptimizerOptions()
    {
        this->lr = learning_rate;
    }

    TORCH_ARG(double, beta1) = 0.9;
    TORCH_ARG(double, beta2) = 0.999;
    TORCH_ARG(double, eps) = 1e-8;
    TORCH_ARG(double, weight_decay) = 0.0;
};

// --- Main Options for GradientCheckpointingOptimizer ---
struct GradientCheckpointingOptions
{
    // We don't inherit from OptimizerOptions because this is a meta-optimizer
    // that doesn't have its own direct hyperparameters like LR.
    std::shared_ptr<torch::nn::Module> model;
    std::function<torch::Tensor(torch::Tensor)> loss_fn;
    std::vector<std::shared_ptr<torch::nn::Module>> checkpointed_modules;
    GCO_InnerOptions inner_optimizer_options;

    GradientCheckpointingOptions(
        std::shared_ptr<torch::nn::Module> _model,
        std::function<torch::Tensor(torch::Tensor)> _loss_fn,
        std::vector<std::shared_ptr<torch::nn::Module>> _checkpointed_modules,
        GCO_InnerOptions _inner_opts = GCO_InnerOptions())
        : model(std::move(_model)),
          loss_fn(std::move(_loss_fn)),
          checkpointed_modules(std::move(_checkpointed_modules)),
          inner_optimizer_options(std::move(_inner_opts))
    {
    }
};

// --- A simplified state, as the real state is in the inner optimizer ---
struct GCParamState : torch::optim::OptimizerParamState
{
    TORCH_ARG(torch::Tensor, step);
    GCParamState() = default;

    std::unique_ptr<OptimizerParamState> clone() const override
    {
        auto cloned = std::make_unique<GCParamState>();
        if (step().defined()) cloned->step(step().clone());
        return cloned;
    }
};

// --- GradientCheckpointingOptimizer Class ---
class GradientCheckpointingOptimizer
{
public:
    // This optimizer does not inherit from torch::optim::Optimizer because its
    // step function has a different signature (it needs input data and targets).
    GradientCheckpointingOptimizer(GradientCheckpointingOptions options);

    // The step function needs data to perform the forward/backward pass.
    torch::Tensor step(const torch::Tensor& input, const torch::Tensor& target);

    // Standard optimizer methods
    void zero_grad();

private:
    std::shared_ptr<torch::nn::Module> model_;
    std::vector<std::shared_ptr<torch::nn::Module>> checkpointed_modules_;
    std::function<torch::Tensor(torch::Tensor)> loss_fn_;
    std::unique_ptr<torch::optim::Adam> inner_optimizer_;
};

#endif // GRADIENT_CHECKPOINTING_OPTIMIZER_HPP
