#ifndef POWERSGD_OPTIMIZER_HPP
#define POWERSGD_OPTIMIZER_HPP

#include <torch/torch.h>
#include <torch/serialize/archive.h>

#include <cmath>
#include <vector>
#include <memory>
#include <string>
#include <cstdint>

// --- Options for PowerSGD Optimizer ---
struct PowerSGDOptions : torch::optim::OptimizerOptions
{
    explicit PowerSGDOptions(double learning_rate = 0.1) // SGD often uses a higher base LR
        : torch::optim::OptimizerOptions()
    {
        this->lr(learning_rate);
    }

    // SGD with Momentum parameters
    TORCH_ARG(double, momentum) = 0.9;
    TORCH_ARG(double, weight_decay) = 1e-4;
    TORCH_ARG(double, lr) = 1e-6;
    // PowerSGD specific parameter
    TORCH_ARG(double, power) = 0.5; // The 'p' in |g|^p.

    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<torch::optim::OptimizerOptions> clone() const override;
};

// --- Parameter State for PowerSGD ---
struct PowerSGDParamState : torch::optim::OptimizerParamState
{
    TORCH_ARG(torch::Tensor, momentum_buffer);

    // PowerSGDParamState() = default;
    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<OptimizerParamState> clone() const override;
};

// --- PowerSGD Optimizer Class ---
class PowerSGD : public torch::optim::Optimizer
{
public:
    PowerSGD(std::vector<torch::Tensor> params, PowerSGDOptions options);
    explicit PowerSGD(std::vector<torch::Tensor> params, double lr = 0.1);

    using LossClosure = std::function<torch::Tensor()>;
    torch::Tensor step(LossClosure closure = nullptr) override;
    void save(torch::serialize::OutputArchive& archive) const override;
    void load(torch::serialize::InputArchive& archive) override;

protected:
    std::unique_ptr<torch::optim::OptimizerParamState> make_param_state() ;
};

#endif // POWERSGD_OPTIMIZER_HPP
