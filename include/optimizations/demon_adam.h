#pragma once


#include "common.h"
namespace xt::optim
{
    // --- Options for DemonAdam Optimizer ---
    struct DemonAdamOptions : torch::optim::OptimizerOptions {
        explicit DemonAdamOptions(double learning_rate = 1e-3)
            : torch::optim::OptimizerOptions() {
            this->lr(learning_rate);
        }

        // Dynamic momentum (beta1) decay parameters
        TORCH_ARG(double, beta1_initial) = 0.99;
        TORCH_ARG(double, beta1_final) = 0.9;
        TORCH_ARG(long, total_steps) = 100000; // Steps over which beta1 decays

        // Standard Adam parameters
        TORCH_ARG(double, beta2) = 0.999;
        TORCH_ARG(double, eps) = 1e-8;
        TORCH_ARG(double, weight_decay) = 0.0;
        TORCH_ARG(double ,  lr) = 1e-6;

        void serialize(torch::serialize::OutputArchive& archive) const override;
        void deserialize(torch::serialize::InputArchive& archive) ;
        std::unique_ptr<torch::optim::OptimizerOptions> clone() const override;
    };

    // --- Parameter State for DemonAdam ---
    struct DemonAdamParamState : torch::optim::OptimizerParamState {
        TORCH_ARG(torch::Tensor, step);
        TORCH_ARG(torch::Tensor, exp_avg);      // m_t
        TORCH_ARG(torch::Tensor, exp_avg_sq);   // v_t

        // DemonAdamParamState() = default;
        void serialize(torch::serialize::OutputArchive& archive) const override;
        void deserialize(torch::serialize::InputArchive& archive) ;
        std::unique_ptr<OptimizerParamState> clone() const override;
    };

    // --- DemonAdam Optimizer Class ---
    class DemonAdam : public torch::optim::Optimizer {
    public:
        DemonAdam(std::vector<torch::Tensor> params, DemonAdamOptions options);
        explicit DemonAdam(std::vector<torch::Tensor> params, double lr = 1e-3);

        using LossClosure = std::function<torch::Tensor()>;
        torch::Tensor step(LossClosure closure = nullptr) override;
        void save(torch::serialize::OutputArchive& archive) const override;
        void load(torch::serialize::InputArchive& archive) override;

    protected:
        std::unique_ptr<torch::optim::OptimizerParamState> make_param_state() ;
    };
}