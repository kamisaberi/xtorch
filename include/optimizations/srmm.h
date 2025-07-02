#pragma once


#include "common.h"
namespace xt::optim
{
    // --- Options for SRMM Optimizer ---
    struct SRMMOptions : torch::optim::OptimizerOptions
    {
        explicit SRMMOptions(double learning_rate = 1e-2)
            : torch::optim::OptimizerOptions()
        {
            this->lr(learning_rate);
        }

        // Momentum EMA decay rate
        TORCH_ARG(double, beta) = 0.9;

        TORCH_ARG(double, weight_decay) = 1e-4;
        TORCH_ARG(double, eps) = 1e-8; // For numerical stability in norm calculations
        TORCH_ARG(double, lr) = 1e-6;
        void serialize(torch::serialize::OutputArchive& archive) const override;
        void deserialize(torch::serialize::InputArchive& archive) ;
        std::unique_ptr<torch::optim::OptimizerOptions> clone() const override;
    };

    // --- Parameter State for SRMM Optimizer ---
    struct SRMMParamState : torch::optim::OptimizerParamState
    {
        TORCH_ARG(torch::Tensor, step);
        TORCH_ARG(torch::Tensor, exp_avg); // m_t (momentum)

        // SRMMParamState() = default;
        void serialize(torch::serialize::OutputArchive& archive) const override;
        void deserialize(torch::serialize::InputArchive& archive) ;
        std::unique_ptr<OptimizerParamState> clone() const override;
    };

    // --- SRMM Optimizer Class ---
    class SRMM : public torch::optim::Optimizer
    {
    public:
        SRMM(std::vector<torch::Tensor> params, SRMMOptions options);
        explicit SRMM(std::vector<torch::Tensor> params, double lr = 1e-2);

        using LossClosure = std::function<torch::Tensor()>;
        torch::Tensor step(LossClosure closure = nullptr) override;
        void save(torch::serialize::OutputArchive& archive) const override;
        void load(torch::serialize::InputArchive& archive) override;

    protected:
        std::unique_ptr<torch::optim::OptimizerParamState> make_param_state() ;
    };
}