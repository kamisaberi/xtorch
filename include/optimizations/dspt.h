#ifndef DSPT_OPTIMIZER_HPP
#define DSPT_OPTIMIZER_HPP

#include <torch/torch.h>
#include <torch/serialize/archive.h>

#include <cmath>
#include <vector>
#include <memory>
#include <string>
#include <cstdint>

// --- Options for DSPT ---
struct DSPTOptions : torch::optim::OptimizerOptions
{
    double lr;

    explicit DSPTOptions(double learning_rate = 1e-3)
        : torch::optim::OptimizerOptions()
    {
        this->lr = learning_rate;
    }

    // Base Adam optimizer parameters
    TORCH_ARG(double, beta1) = 0.9;
    TORCH_ARG(double, beta2) = 0.999;
    TORCH_ARG(double, eps) = 1e-8;
    TORCH_ARG(double, weight_decay) = 0.0;

    // Sparsity parameters
    TORCH_ARG(double, sparsity) = 0.9; // Target sparsity (e.g., 90% of weights are zero)
    TORCH_ARG(double, prune_rate) = 0.1; // Fraction of *dense* weights to prune/grow at each update
    TORCH_ARG(long, prune_frequency) = 1000;
    TORCH_ARG(long, start_pruning_step) = 2000;

    // Low-rank update parameters
    TORCH_ARG(int, low_rank_k) = 1; // Rank of the low-rank update. k=0 disables this feature.

    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive);
    std::unique_ptr<torch::optim::OptimizerOptions> clone() const override;
};

// --- Parameter State for DSPT ---
struct DSPTParamState : torch::optim::OptimizerParamState
{
    TORCH_ARG(torch::Tensor, step);
    TORCH_ARG(torch::Tensor, exp_avg); // m_t (momentum)
    TORCH_ARG(torch::Tensor, exp_avg_sq); // v_t (variance)
    TORCH_ARG(torch::Tensor, mask); // The crucial sparsity mask

    DSPTParamState() = default;
    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive);
    std::unique_ptr<OptimizerParamState> clone() const override;
};

// --- DSPT Optimizer Class ---
class DSPT : public torch::optim::Optimizer
{
public:
    DSPT(std::vector<torch::Tensor> params, DSPTOptions options);

    using LossClosure = std::function<torch::Tensor()>;
    torch::Tensor step(LossClosure closure = nullptr) override;
    void save(torch::serialize::OutputArchive& archive) const override;
    void load(torch::serialize::InputArchive& archive) override;

protected:
    std::unique_ptr<torch::optim::OptimizerParamState> make_param_state();

private:
    void _update_mask(
        torch::Tensor& param,
        const torch::Tensor& full_grad,
        DSPTParamState& state,
        const DSPTOptions& options);

    void _apply_low_rank_update(
        torch::Tensor& param,
        DSPTParamState& state,
        const DSPTOptions& options);
};

#endif // DSPT_OPTIMIZER_HPP
