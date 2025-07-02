#pragma once


#include "common.h"
namespace xt::optim
{
    // --- Options for Gravity Optimizer ---
    struct GravityOptions : torch::optim::OptimizerOptions {
        explicit GravityOptions(double learning_rate = 1e-3)
            : torch::optim::OptimizerOptions() {
            this->lr(learning_rate);
        }

        // Adam-like parameters
        TORCH_ARG(double, beta1) = 0.9;   // For momentum on the gradient "force"
        TORCH_ARG(double, beta2) = 0.999; // For the curvature estimate (v_t)
        TORCH_ARG(double, eps) = 1e-8;
        TORCH_ARG(double ,  lr) = 1e-6;   // For momentum on the gradient "force"

        // "Gravitational" regularization parameter (similar to weight decay strength)
        TORCH_ARG(double, gravitational_constant) = 1e-4;

        void serialize(torch::serialize::OutputArchive& archive) const override;
        void deserialize(torch::serialize::InputArchive& archive) ;
        std::unique_ptr<torch::optim::OptimizerOptions> clone() const override;
    };

    // --- Parameter State for Gravity Optimizer ---
    struct GravityParamState : torch::optim::OptimizerParamState {
        TORCH_ARG(torch::Tensor, step);
        TORCH_ARG(torch::Tensor, exp_avg);      // m_t (Momentum on the gradient force)
        TORCH_ARG(torch::Tensor, exp_avg_sq);   // v_t (Curvature of the loss landscape)

        // GravityParamState() = default;
        void serialize(torch::serialize::OutputArchive& archive) const override;
        void deserialize(torch::serialize::InputArchive& archive) ;
        std::unique_ptr<OptimizerParamState> clone() const override;
    };

    // --- Gravity Optimizer Class ---
    class Gravity : public torch::optim::Optimizer {
    public:
        Gravity(std::vector<torch::Tensor> params, GravityOptions options);

        using LossClosure = std::function<torch::Tensor()>;
        torch::Tensor step(LossClosure closure = nullptr) override;
        void save(torch::serialize::OutputArchive& archive) const override;
        void load(torch::serialize::InputArchive& archive) override;

    protected:
        std::unique_ptr<torch::optim::OptimizerParamState> make_param_state() ;
    };
}