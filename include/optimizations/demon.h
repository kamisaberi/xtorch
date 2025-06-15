#ifndef DEMON_OPTIMIZER_HPP
#define DEMON_OPTIMIZER_HPP

#include <torch/torch.h>
#include <torch/serialize/archive.h>

#include <cmath>
#include <vector>
#include <memory>
#include <string>
#include <cstdint>

// --- Options for Demon Optimizer ---
struct DemonOptions : torch::optim::OptimizerOptions {
    explicit DemonOptions(double learning_rate = 0.1) // SGD-like LR
        : torch::optim::OptimizerOptions() {
        this->lr(learning_rate);
    }

    // The momentum will decay from beta_initial to beta_final
    TORCH_ARG(double, beta_initial) = 0.99;
    TORCH_ARG(double, beta_final) = 0.5;

    // The total number of steps over which the decay occurs
    TORCH_ARG(long, total_steps) = 100000;

    TORCH_ARG(double, weight_decay) = 1e-4;
    TORCH_ARG(double ,  lr) = 1e-6;

    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<torch::optim::OptimizerOptions> clone() const override;
};

// --- Parameter State for Demon Optimizer ---
struct DemonParamState : torch::optim::OptimizerParamState {
    TORCH_ARG(torch::Tensor, step);
    TORCH_ARG(torch::Tensor, momentum_buffer);

    // DemonParamState() = default;
    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<OptimizerParamState> clone() const override;
};

// --- Demon Optimizer Class ---
class Demon : public torch::optim::Optimizer {
public:
    Demon(std::vector<torch::Tensor> params, DemonOptions options);
    explicit Demon(std::vector<torch::Tensor> params, double lr = 0.1);

    using LossClosure = std::function<torch::Tensor()>;
    torch::Tensor step(LossClosure closure = nullptr) override;
    void save(torch::serialize::OutputArchive& archive) const override;
    void load(torch::serialize::InputArchive& archive) override;

protected:
    std::unique_ptr<torch::optim::OptimizerParamState> make_param_state() ;
};

#endif // DEMON_OPTIMIZER_HPP