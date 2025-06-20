#pragma once


#include "common.h"
namespace xt::optim
{
    // --- Options for HGS Optimizer ---
    struct HGSOptions : torch::optim::OptimizerOptions {

        explicit HGSOptions(double learning_rate = 1e-3)
            : torch::optim::OptimizerOptions() {
            this->lr(learning_rate);
        }

        // Base Adam optimizer parameters
        TORCH_ARG(double, beta1) = 0.9;
        TORCH_ARG(double, beta2) = 0.999;
        TORCH_ARG(double, eps) = 1e-8;
        TORCH_ARG(double, weight_decay) = 0.0;
        TORCH_ARG(double ,  lr) = 1e-6;   // For momentum on the gradient "force"

        // Sparsification and Sampling parameters
        TORCH_ARG(double, compression_ratio) = 0.01; // Select top 1% of gradients
        TORCH_ARG(double, sampling_beta) = 0.99;     // EMA decay for adaptive sampling scores

        void serialize(torch::serialize::OutputArchive& archive) const override;
        void deserialize(torch::serialize::InputArchive& archive) ;
        std::unique_ptr<torch::optim::OptimizerOptions> clone() const override;
    };

    // --- Parameter State for HGS ---
    struct HGSParamState : torch::optim::OptimizerParamState {
        TORCH_ARG(torch::Tensor, step);
        TORCH_ARG(torch::Tensor, error_feedback);   // Error accumulation
        TORCH_ARG(torch::Tensor, sampling_scores);  // Adaptive scores for selection

        // States for the inner Adam optimizer
        TORCH_ARG(torch::Tensor, exp_avg);      // m_t
        TORCH_ARG(torch::Tensor, exp_avg_sq);   // v_t

        // HGSParamState() = default;
        void serialize(torch::serialize::OutputArchive& archive) const override;
        void deserialize(torch::serialize::InputArchive& archive) ;
        std::unique_ptr<OptimizerParamState> clone() const override;
    };

    // --- HGS Optimizer Class ---
    class HGS : public torch::optim::Optimizer {
    public:
        HGS(std::vector<torch::Tensor> params, HGSOptions options);

        using LossClosure = std::function<torch::Tensor()>;
        torch::Tensor step(LossClosure closure = nullptr) override;
        void save(torch::serialize::OutputArchive& archive) const override;
        void load(torch::serialize::InputArchive& archive) override;

    protected:
        std::unique_ptr<torch::optim::OptimizerParamState> make_param_state() ;
    };
}