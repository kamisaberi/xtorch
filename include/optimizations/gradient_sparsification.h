#pragma once


#include "common.h"
namespace xt::optim
{
    // --- Options for GradientSparsification Optimizer ---
    struct GradientSparsificationOptions : torch::optim::OptimizerOptions {
        double lr;
        explicit GradientSparsificationOptions(double learning_rate = 0.1) // SGD often uses a higher LR
            : torch::optim::OptimizerOptions() {
            this->lr= learning_rate;
        }

        // Base SGD with Momentum parameters
        TORCH_ARG(double, momentum) = 0.9;
        TORCH_ARG(double, weight_decay) = 1e-4;

        // Sparsification parameters
        TORCH_ARG(double, compression_ratio) = 0.01; // Select top 1% of gradients.
        // TORCH_ARG(bool, warmup) = true; // Optional: start dense and begin sparsifying after some steps

        void serialize(torch::serialize::OutputArchive& archive) const override;
        void deserialize(torch::serialize::InputArchive& archive) ;
        std::unique_ptr<torch::optim::OptimizerOptions> clone() const override;
    };

    // --- Parameter State for GradientSparsification ---
    struct GSParamState : torch::optim::OptimizerParamState {
        TORCH_ARG(torch::Tensor, step);
        TORCH_ARG(torch::Tensor, momentum_buffer); // For inner SGD
        TORCH_ARG(torch::Tensor, error_feedback);  // Error accumulation

        // GSParamState() = default;
        void serialize(torch::serialize::OutputArchive& archive) const override;
        void deserialize(torch::serialize::InputArchive& archive) ;
        std::unique_ptr<OptimizerParamState> clone() const override;
    };

    // --- GradientSparsification Optimizer Class ---
    class GradientSparsification : public torch::optim::Optimizer {
    public:
        GradientSparsification(std::vector<torch::Tensor> params, GradientSparsificationOptions options);

        using LossClosure = std::function<torch::Tensor()>;
        torch::Tensor step(LossClosure closure = nullptr) override;
        void save(torch::serialize::OutputArchive& archive) const override;
        void load(torch::serialize::InputArchive& archive) override;

    protected:
        std::unique_ptr<torch::optim::OptimizerParamState> make_param_state() ;
    };
}