#pragma once


#include "common.h"
namespace xt::optim
{
    // --- Options for PLO (Projected Lookahead Optimizer) ---
    struct PLOOptions :public torch::optim::OptimizerOptions {
    public:
        explicit PLOOptions(double learning_rate = 1.0) // Both Lookahead and Projection are self-tuning
            : torch::optim::OptimizerOptions() {
            this->lr(learning_rate);
        }

        // Lookahead parameters
        TORCH_ARG(double, alpha) = 0.5; // Interpolation factor for slow weights
        TORCH_ARG(long, k) = 6;         // Sync/update frequency
        TORCH_ARG(double, lr) = 1e-6;

        // Inner Projected Optimizer parameters
        TORCH_ARG(double, beta_momentum) = 0.9; // Momentum on the projected update
        TORCH_ARG(double, beta_stats) = 0.999;  // EMA decay for Kronecker factor statistics
        TORCH_ARG(double, damping) = 1e-6;      // Damping for matrix inverse root
        TORCH_ARG(double, weight_decay) = 1e-4;
        TORCH_ARG(int, root_order) = 2;
        TORCH_ARG(long, precondition_frequency) = 20; // Update preconditioner more frequently for fast weights

        void serialize(torch::serialize::OutputArchive& archive) const override;
        void deserialize(torch::serialize::InputArchive& archive) ;
        std::unique_ptr<torch::optim::OptimizerOptions> clone() const override;
    };

    // --- Parameter State for PLO ---
    struct PLOParamState :public torch::optim::OptimizerParamState {
        TORCH_ARG(torch::Tensor, step);

        // Lookahead state
        TORCH_ARG(torch::Tensor, slow_param);

        // Inner Projected Optimizer state
        TORCH_ARG(torch::Tensor, momentum_buffer);
    public:
        torch::Tensor l_ema; // Left factor
        torch::Tensor r_ema; // Right factor
        torch::Tensor l_inv_root;
        torch::Tensor r_inv_root;

        // PLOParamState() = default;
        void serialize(torch::serialize::OutputArchive& archive) const override;
        void deserialize(torch::serialize::InputArchive& archive) ;
        std::unique_ptr<OptimizerParamState> clone() const override;
    };

    // --- PLO Optimizer Class ---
    class PLO : public torch::optim::Optimizer {
    public:
        PLO(std::vector<torch::Tensor> params, PLOOptions options);

        using LossClosure = std::function<torch::Tensor()>;
        torch::Tensor step(LossClosure closure = nullptr) override;
        void save(torch::serialize::OutputArchive& archive) const override;
        void load(torch::serialize::InputArchive& archive) override;

    protected:
        std::unique_ptr<torch::optim::OptimizerParamState> make_param_state() ;

    private:
        void _fallback_to_sgd(
            torch::Tensor& param,
            const torch::Tensor& grad,
            PLOParamState& state,
            const PLOOptions& options);

        torch::Tensor _compute_matrix_inverse_root(
            const torch::Tensor& matrix,
            double damping,
            int root_order);
    };
}