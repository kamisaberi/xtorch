#pragma once


#include "common.h"
namespace xt::optim
{
    // --- Options for SLAMB Optimizer ---
    struct SLAMBOptions : torch::optim::OptimizerOptions {
        explicit SLAMBOptions(double learning_rate = 1e-3)
            : torch::optim::OptimizerOptions() {
            this->lr(learning_rate);
        }

        // LAMB parameters
        TORCH_ARG(double, beta1) = 0.9;
        TORCH_ARG(double, beta2) = 0.999;
        TORCH_ARG(double, eps) = 1e-6;
        TORCH_ARG(double, weight_decay) = 0.01;
        TORCH_ARG(double, lr) = 1e-6;
        // Sparsification parameter
        TORCH_ARG(double, compression_ratio) = 0.01; // Select top 1% of gradients

        void serialize(torch::serialize::OutputArchive& archive) const override;
        void deserialize(torch::serialize::InputArchive& archive) ;
        std::unique_ptr<torch::optim::OptimizerOptions> clone() const override;
    };

    // --- Parameter State for SLAMB Optimizer ---
    struct SLAMBParamState : torch::optim::OptimizerParamState {
        TORCH_ARG(torch::Tensor, step);
        TORCH_ARG(torch::Tensor, exp_avg);      // m_t
        TORCH_ARG(torch::Tensor, exp_avg_sq);   // v_t
        TORCH_ARG(torch::Tensor, error_feedback); // For sparsification

        // SLAMBParamState() = default;
        void serialize(torch::serialize::OutputArchive& archive) const override;
        void deserialize(torch::serialize::InputArchive& archive) ;
        std::unique_ptr<OptimizerParamState> clone() const override;
    };

    // --- SLAMB Optimizer Class ---
    class SLAMB : public torch::optim::Optimizer {
    public:
        SLAMB(std::vector<torch::Tensor> params, SLAMBOptions options);
        explicit SLAMB(std::vector<torch::Tensor> params, double lr = 1e-3);

        using LossClosure = std::function<torch::Tensor()>;
        torch::Tensor step(LossClosure closure = nullptr) override;
        void save(torch::serialize::OutputArchive& archive) const override;
        void load(torch::serialize::InputArchive& archive) override;

    protected:
        std::unique_ptr<torch::optim::OptimizerParamState> make_param_state() ;
    };
}