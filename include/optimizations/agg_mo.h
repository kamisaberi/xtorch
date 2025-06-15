#ifndef AGGMO_OPTIMIZER_HPP
#define AGGMO_OPTIMIZER_HPP

#include <torch/torch.h>
#include <torch/serialize/archive.h>

#include <cmath>
#include <vector>
#include <memory>
#include <string>
#include <cstdint>

// --- Options for AggMo Optimizer ---
struct AggMoOptions :public torch::optim::OptimizerOptions {
    explicit AggMoOptions(double learning_rate = 0.1) // Often uses SGD-like LRs
        : torch::optim::OptimizerOptions() {
        this->lr(learning_rate);
        // Default betas for AggMo (from the paper)
        betas_ = {0.0, 0.9, 0.99};
    }
    TORCH_ARG(double ,  lr) = 1e-6;
    // Vector of beta values for each momentum buffer
    // This is not a TORCH_ARG directly, as it's a std::vector.
    // We handle its serialization/deserialization manually.
public:

    std::vector<double> betas_;
    AggMoOptions& betas(const std::vector<double>& new_betas) {
        betas_ = new_betas;
        return *this;
    }
    const std::vector<double>& betas() const { return betas_; }


    TORCH_ARG(double, weight_decay) = 0.0;

    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<torch::optim::OptimizerOptions> clone() const override;
};

// --- Parameter State for AggMo Optimizer ---
struct AggMoParamState : public torch::optim::OptimizerParamState {
    // A vector of momentum buffers, one for each beta
    std::vector<torch::Tensor> momentum_buffers;
public:
    // AggMoParamState() = default;
    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<OptimizerParamState> clone() const override;
};

// --- AggMo Optimizer Class ---
class AggMo : public torch::optim::Optimizer {
public:
    AggMo(std::vector<torch::Tensor> params, AggMoOptions options);
    explicit AggMo(std::vector<torch::Tensor> params, double lr = 0.1);

    using LossClosure = std::function<torch::Tensor()>;
    torch::Tensor step(LossClosure closure = nullptr) override;
    void save(torch::serialize::OutputArchive& archive) const override;
    void load(torch::serialize::InputArchive& archive) override;

protected:
    std::unique_ptr<torch::optim::OptimizerParamState> make_param_state() ;
};

#endif // AGGMO_OPTIMIZER_HPP