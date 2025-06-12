#ifndef LOOKAHEAD_OPTIMIZER_HPP
#define LOOKAHEAD_OPTIMIZER_HPP

#include <torch/torch.h>
#include <torch/serialize/archive.h>

#include <cmath>
#include <vector>
#include <memory>
#include <string>
#include <cstdint>
#include <functional>

// --- Options for the inner Adam optimizer to be wrapped ---
struct LookaheadInnerAdamOptions :public torch::optim::OptimizerOptions {
public:
    explicit LookaheadInnerAdamOptions(double learning_rate = 1e-3)
        : torch::optim::OptimizerOptions() {
        this->lr(learning_rate);
    }
    TORCH_ARG(double ,  lr) = 1e-6;   // For momentum on the gradient "force"
    TORCH_ARG(double, beta1) = 0.9;
    TORCH_ARG(double, beta2) = 0.999;
    TORCH_ARG(double, eps) = 1e-8;
    TORCH_ARG(double, weight_decay) = 0.0;
};

// --- Main Options for Lookahead Optimizer ---
struct LookaheadOptions : torch::optim::OptimizerOptions {
public:
    // Note: LR in base class will be for the inner optimizer.
    explicit LookaheadOptions(double learning_rate = 1e-3)
        : torch::optim::OptimizerOptions() {
            this->lr(learning_rate);
        }

    TORCH_ARG(double ,  lr) = 1e-6;   // For momentum on the gradient "force"
    // Lookahead-specific parameters
    TORCH_ARG(double, alpha) = 0.5; // Interpolation factor for slow weights.
    TORCH_ARG(long, k) = 6;         // Sync/update frequency for slow weights.

    // Inner Adam optimizer parameters are included for convenience
    TORCH_ARG(double, beta1) = 0.9;
    TORCH_ARG(double, beta2) = 0.999;
    TORCH_ARG(double, eps) = 1e-8;
    TORCH_ARG(double, weight_decay) = 0.0;

    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<torch::optim::OptimizerOptions> clone() const override;
};

// --- Parameter State for Lookahead ---
struct LookaheadParamState : torch::optim::OptimizerParamState {
    TORCH_ARG(torch::Tensor, step);

    // Slow weights: the stable, main parameters for lookahead
    TORCH_ARG(torch::Tensor, slow_param);

    // Inner Adam optimizer states
    TORCH_ARG(torch::Tensor, exp_avg);      // m_t
    TORCH_ARG(torch::Tensor, exp_avg_sq);   // v_t

    LookaheadParamState() = default;
    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<OptimizerParamState> clone() const override;
};

// --- Lookahead Optimizer Class ---
class Lookahead : public torch::optim::Optimizer {
public:
    Lookahead(std::vector<torch::Tensor> params, LookaheadOptions options);

    using LossClosure = std::function<torch::Tensor()>;
    torch::Tensor step(LossClosure closure = nullptr) override;
    void save(torch::serialize::OutputArchive& archive) const override;
    void load(torch::serialize::InputArchive& archive) override;

protected:
    std::unique_ptr<torch::optim::OptimizerParamState> make_param_state() ;
};

#endif // LOOKAHEAD_OPTIMIZER_HPP