#ifndef PO_OPTIMIZER_HPP
#define PO_OPTIMIZER_HPP

#include <torch/torch.h>
#include <torch/serialize/archive.h>

#include <cmath>
#include <vector>
#include <memory>
#include <string>
#include <cstdint>

// --- Options for PO (Projected Optimizer) ---
struct POOptions : torch::optim::OptimizerOptions {
    explicit POOptions(double learning_rate = 1.0) // Projection provides a dynamic LR scale
        : torch::optim::OptimizerOptions() {
        this->lr(learning_rate);
    }

    // Momentum parameter
    TORCH_ARG(double, beta) = 0.9; // For the historical gradient EMA (momentum)

    TORCH_ARG(double, weight_decay) = 1e-4;
    TORCH_ARG(double, eps) = 1e-8; // For numerical stability in projection denominator
    TORCH_ARG(double, lr) = 1e-6;
    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<torch::optim::OptimizerOptions> clone() const override;
};

// --- Parameter State for PO Optimizer ---
struct POParamState : torch::optim::OptimizerParamState {
    TORCH_ARG(torch::Tensor, step);
    TORCH_ARG(torch::Tensor, exp_avg); // m_t (momentum)

    POParamState() = default;
    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<OptimizerParamState> clone() const override;
};

// --- PO Optimizer Class ---
class PO : public torch::optim::Optimizer {
public:
    PO(std::vector<torch::Tensor> params, POOptions options);
    explicit PO(std::vector<torch::Tensor> params, double lr = 1.0);

    using LossClosure = std::function<torch::Tensor()>;
    torch::Tensor step(LossClosure closure = nullptr) override;
    void save(torch::serialize::OutputArchive& archive) const override;
    void load(torch::serialize::InputArchive& archive) override;

protected:
    std::unique_ptr<torch::optim::OptimizerParamState> make_param_state() ;
};

#endif // PO_OPTIMIZER_HPP