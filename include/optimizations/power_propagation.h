#pragma once


#include "common.h"
namespace xt::optim
{
    // --- Options for PowerPropagation Optimizer ---
    struct PowerPropOptions : torch::optim::OptimizerOptions {
        explicit PowerPropOptions(double learning_rate = 1e-3)
            : torch::optim::OptimizerOptions() {
            this->lr(learning_rate);
        }

        // Inner Adam parameters
        TORCH_ARG(double, beta1) = 0.9;
        TORCH_ARG(double, beta2) = 0.999;
        TORCH_ARG(double, eps) = 1e-8;
        TORCH_ARG(double, weight_decay) = 0.0;
        TORCH_ARG(double, lr) = 1e-6;

        // PowerPropagation specific parameter
        TORCH_ARG(double, power) = 0.5; // The 'p' in |g|^p. p < 1 is dampening, p > 1 is accelerating.

        void serialize(torch::serialize::OutputArchive& archive) const override;
        void deserialize(torch::serialize::InputArchive& archive) ;
        std::unique_ptr<torch::optim::OptimizerOptions> clone() const override;
    };

    // --- Parameter State for PowerPropagation ---
    struct PowerPropParamState : torch::optim::OptimizerParamState {
        TORCH_ARG(torch::Tensor, step);
        TORCH_ARG(torch::Tensor, exp_avg);      // m_t (EMA of powered gradients)
        TORCH_ARG(torch::Tensor, exp_avg_sq);   // v_t (EMA of squared powered gradients)

        // PowerPropParamState() = default;
        void serialize(torch::serialize::OutputArchive& archive) const override;
        void deserialize(torch::serialize::InputArchive& archive) ;
        std::unique_ptr<OptimizerParamState> clone() const override;
    };

    // --- PowerPropagation Optimizer Class ---
    class PowerPropagation : public torch::optim::Optimizer {
    public:
        PowerPropagation(std::vector<torch::Tensor> params, PowerPropOptions options);
        explicit PowerPropagation(std::vector<torch::Tensor> params, double lr = 1e-3);

        using LossClosure = std::function<torch::Tensor()>;
        torch::Tensor step(LossClosure closure = nullptr) override;
        void save(torch::serialize::OutputArchive& archive) const override;
        void load(torch::serialize::InputArchive& archive) override;

    protected:
        std::unique_ptr<torch::optim::OptimizerParamState> make_param_state() ;
    };
}