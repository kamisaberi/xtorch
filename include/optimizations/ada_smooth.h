#pragma once


#include "common.h"
namespace xt::optim
{
    // --- Options for AdaSmooth Optimizer ---
    struct AdaSmoothOptions : torch::optim::OptimizerOptions {
        explicit AdaSmoothOptions(double learning_rate = 1e-3)
            : torch::optim::OptimizerOptions() {
            this->lr(learning_rate);
        }

        // Adam parameters
        TORCH_ARG(double, beta1) = 0.9;
        TORCH_ARG(double, beta2) = 0.999;

        // Smoothing parameter for the final update step
        TORCH_ARG(double, beta3) = 0.9; // A second layer of momentum

        TORCH_ARG(double, eps) = 1e-8;
        TORCH_ARG(double, weight_decay) = 0.0;
        TORCH_ARG(double ,  lr) = 1e-6;

        void serialize(torch::serialize::OutputArchive& archive) const override;
        void deserialize(torch::serialize::InputArchive& archive) ;
        std::unique_ptr<torch::optim::OptimizerOptions> clone() const override;
    };

    // --- Parameter State for AdaSmooth Optimizer ---
    struct AdaSmoothParamState : torch::optim::OptimizerParamState {
        TORCH_ARG(torch::Tensor, step);
        TORCH_ARG(torch::Tensor, exp_avg);      // m_t (Adam's first moment)
        TORCH_ARG(torch::Tensor, exp_avg_sq);   // v_t (Adam's second moment)
        TORCH_ARG(torch::Tensor, smooth_update); // s_t (EMA of final update steps)

        // AdaSmoothParamState() = default;
        void serialize(torch::serialize::OutputArchive& archive) const override;
        void deserialize(torch::serialize::InputArchive& archive) ;
        std::unique_ptr<OptimizerParamState> clone() const override;
    };

    // --- AdaSmooth Optimizer Class ---
    class AdaSmooth : public torch::optim::Optimizer {
    public:
        AdaSmooth(std::vector<torch::Tensor> params, AdaSmoothOptions options);
        explicit AdaSmooth(std::vector<torch::Tensor> params, double lr = 1e-3);

        using LossClosure = std::function<torch::Tensor()>;
        torch::Tensor step(LossClosure closure = nullptr) override;
        void save(torch::serialize::OutputArchive& archive) const override;
        void load(torch::serialize::InputArchive& archive) override;

    protected:
        std::unique_ptr<torch::optim::OptimizerParamState> make_param_state() ;
    };
}