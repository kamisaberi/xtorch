#pragma once


#include "common.h"
namespace xt::optim
{
    // --- Options for YellowFin Optimizer ---
    // YellowFin is mostly self-tuning, so it has fewer user-facing knobs.
    struct YellowFinOptions : torch::optim::OptimizerOptions {
        explicit YellowFinOptions(double learning_rate = 1.0) // Initial LR, will be overridden
            : torch::optim::OptimizerOptions() {
            this->lr(learning_rate);
        }

        // EMA decay for smoothing the measurements
        TORCH_ARG(double, beta) = 0.999;
        TORCH_ARG(double, weight_decay) = 0.0;
        // Curvature range clipping for stability
        TORCH_ARG(double, curv_min_clamp) = 1e-6;
        TORCH_ARG(double, curv_max_clamp) = 1e6;
        TORCH_ARG(double, lr) = 1e-6;

        void serialize(torch::serialize::OutputArchive& archive) const override;
        void deserialize(torch::serialize::InputArchive& archive) ;
        std::unique_ptr<torch::optim::OptimizerOptions> clone() const override;
    };

    // --- Parameter State for YellowFin Optimizer ---
    // State is global, not per-parameter, which is unique.
    // We will store it in the first parameter's state dictionary.
    struct YellowFinGlobalState {
        // Smoothed measurements
        double h_min_ema = 0.0;
        double h_max_ema = 0.0;
        double grad_var_ema = 0.0;

        // Tuned hyperparameters
        double tuned_lr = 1.0;
        double tuned_momentum = 0.0;

        // For calculating curvature
        torch::Tensor prev_grad_flat;
        torch::Tensor prev_param_flat;
    };

    // --- Per-parameter state is just momentum ---
    struct YellowFinParamState : torch::optim::OptimizerParamState {
        TORCH_ARG(torch::Tensor, momentum_buffer);

        // YellowFinParamState() = default;
        void serialize(torch::serialize::OutputArchive& archive) const override;
        void deserialize(torch::serialize::InputArchive& archive) ;
        std::unique_ptr<OptimizerParamState> clone() const override;
    };

    // --- YellowFin Optimizer Class ---
    class YellowFin : public torch::optim::Optimizer {
    public:
        YellowFin(std::vector<torch::Tensor> params, YellowFinOptions options);

        using LossClosure = std::function<torch::Tensor()>;
        torch::Tensor step(LossClosure closure = nullptr) override;
        void save(torch::serialize::OutputArchive& archive) const override;
        void load(torch::serialize::InputArchive& archive) override;

    protected:
        std::unique_ptr<torch::optim::OptimizerParamState> make_param_state() ;

    private:
        YellowFinGlobalState global_state_; // All tuning happens globally
        long step_count_ = 0;
    };
}