#ifndef GCANS_OPTIMIZER_HPP
#define GCANS_OPTIMIZER_HPP

#include <torch/torch.h>
#include <torch/serialize/archive.h>

#include <cmath>
#include <vector>
#include <memory>
#include <string>
#include <cstdint>

// --- Options for GCANS Optimizer ---
struct GCANSOptions : torch::optim::OptimizerOptions {
    double lr;
    explicit GCANSOptions(double learning_rate = 0.1) // SGD often uses a higher LR
        : torch::optim::OptimizerOptions() {
        this->lr= learning_rate;
    }

    // Base SGD with Momentum parameters
    TORCH_ARG(double, momentum) = 0.9;
    TORCH_ARG(double, weight_decay) = 1e-4;

    // GCANS parameters
    TORCH_ARG(double, compression_ratio) = 0.01; // Select top 1% of gradients
    TORCH_ARG(double, sampling_beta) = 0.99; // EMA decay for sampling probabilities

    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<torch::optim::OptimizerOptions> clone() const override;
};

// --- Parameter State for GCANS ---
struct GCANSParamState : torch::optim::OptimizerParamState {
    TORCH_ARG(torch::Tensor, step);
    TORCH_ARG(torch::Tensor, momentum_buffer); // For inner SGD
    TORCH_ARG(torch::Tensor, error_feedback);  // Error accumulation
    TORCH_ARG(torch::Tensor, sampling_probs);  // Adaptive sampling probabilities
public:
    // GCANSParamState() = default;
    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<OptimizerParamState> clone() const override;
};

// --- GCANS Optimizer Class ---
class GCANS : public torch::optim::Optimizer {
public:
    GCANS(std::vector<torch::Tensor> params, GCANSOptions options);

    using LossClosure = std::function<torch::Tensor()>;
    torch::Tensor step(LossClosure closure = nullptr) override;
    void save(torch::serialize::OutputArchive& archive) const override;
    void load(torch::serialize::InputArchive& archive) override;

protected:
    std::unique_ptr<torch::optim::OptimizerParamState> make_param_state() ;
};

#endif // GCANS_OPTIMIZER_HPP