#pragma once


#include "common.h"
namespace xt::optim
{
    // --- Options for LARS Optimizer ---
    struct LARSOptions : torch::optim::OptimizerOptions {
        explicit LARSOptions(double learning_rate = 0.1)
            : torch::optim::OptimizerOptions() {
            this->lr(learning_rate);
        }

        TORCH_ARG(double, momentum) = 0.9;
        TORCH_ARG(double, weight_decay) = 1e-4;
        TORCH_ARG(double, trust_coefficient) = 0.001;
        TORCH_ARG(double, eps) = 1e-8;
        TORCH_ARG(double, lr) = 1e-6;

        void serialize(torch::serialize::OutputArchive& archive) const override;
        void deserialize(torch::serialize::InputArchive& archive) ;
        std::unique_ptr<torch::optim::OptimizerOptions> clone() const override;
    };

    // --- Parameter State for LARS Optimizer ---
    struct LARSParamState : torch::optim::OptimizerParamState {
        TORCH_ARG(torch::Tensor, momentum_buffer);

        // LARSParamState() = default;
        void serialize(torch::serialize::OutputArchive& archive) const override;
        void deserialize(torch::serialize::InputArchive& archive) ;
        std::unique_ptr<OptimizerParamState> clone() const override;
    };

    // --- LARS Optimizer Class ---
    class LARS : public torch::optim::Optimizer {
    public:
        LARS(std::vector<torch::Tensor> params, LARSOptions options);
        explicit LARS(std::vector<torch::Tensor> params, double lr = 0.1);

        using LossClosure = std::function<torch::Tensor()>;
        torch::Tensor step(LossClosure closure = nullptr) override;
        void save(torch::serialize::OutputArchive& archive) const override;
        void load(torch::serialize::InputArchive& archive) override;

    protected:
        std::unique_ptr<torch::optim::OptimizerParamState> make_param_state() ;
    };
}