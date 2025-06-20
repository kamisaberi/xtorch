#pragma once


#include "common.h"
namespace xt::optim
{
    // --- Options for INFO Optimizer ---
    struct InfoOptions : torch::optim::OptimizerOptions {
        explicit InfoOptions(double learning_rate = 1e-3)
            : torch::optim::OptimizerOptions() {
            this->lr(learning_rate);
        }

        // Adam-like momentum and adaptive LR parameters
        TORCH_ARG(double, beta1) = 0.9;   // Momentum on the natural gradient
        TORCH_ARG(double, beta2) = 0.999; // EMA for the adaptive LR (v_t)

        // Fisher Information parameters
        TORCH_ARG(double, fisher_beta) = 0.999; // EMA for the diagonal FIM estimate
        TORCH_ARG(double, fisher_damping) = 1e-4; // Damping for FIM inverse

        TORCH_ARG(double, eps) = 1e-8;
        TORCH_ARG(double, weight_decay) = 0.0;
        TORCH_ARG(double ,  lr) = 1e-6;   // For momentum on the gradient "force"

        void serialize(torch::serialize::OutputArchive& archive) const override;
        void deserialize(torch::serialize::InputArchive& archive) ;
        std::unique_ptr<torch::optim::OptimizerOptions> clone() const override;
    };

    // --- Parameter State for INFO Optimizer ---
    struct InfoParamState : torch::optim::OptimizerParamState {
        TORCH_ARG(torch::Tensor, step);

        // Diagonal Fisher Information Matrix estimate
        TORCH_ARG(torch::Tensor, fisher_diag_ema);

        // Adam states for the final update
        TORCH_ARG(torch::Tensor, exp_avg);      // m_t (momentum on natural gradient)
        TORCH_ARG(torch::Tensor, exp_avg_sq);   // v_t (adaptive LR on natural gradient)

        // InfoParamState() = default;
        void serialize(torch::serialize::OutputArchive& archive) const override;
        void deserialize(torch::serialize::InputArchive& archive) ;
        std::unique_ptr<OptimizerParamState> clone() const override;
    };

    // --- INFO Optimizer Class ---
    class INFO : public torch::optim::Optimizer {
    public:
        INFO(std::vector<torch::Tensor> params, InfoOptions options);

        using LossClosure = std::function<torch::Tensor()>;
        torch::Tensor step(LossClosure closure = nullptr) override;
        void save(torch::serialize::OutputArchive& archive) const override;
        void load(torch::serialize::InputArchive& archive) override;

    protected:
        std::unique_ptr<torch::optim::OptimizerParamState> make_param_state() ;
    };
}