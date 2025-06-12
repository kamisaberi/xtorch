#ifndef DIAG_ADA_FISHER_HPP
#define DIAG_ADA_FISHER_HPP

#include <torch/torch.h>
#include <torch/serialize/archive.h>

#include <cmath>
#include <vector>
#include <memory>
#include <string>
#include <cstdint> // For int64_t

// --- Options for DiagonalAdaFisher ---
struct DiagonalAdaFisherOptions : torch::optim::OptimizerOptions
{
    double lr;

    explicit DiagonalAdaFisherOptions(double learning_rate = 1e-2) // RMSProp/AdaGrad often use higher LR
        : torch::optim::OptimizerOptions()
    {
        this->lr=learning_rate;
    }

    // Beta for the EMA of squared gradients (diagonal Fisher estimate)
    TORCH_ARG(double, beta) = 0.99; // Common default for RMSProp-like decay
    TORCH_ARG(double, eps) = 1e-8; // Damping term
    TORCH_ARG(double, weight_decay) = 0.0;
    // Note: True Fisher methods might have distinct learning rates for different parts
    // or more complex damping schedules. This is a simplified version.

    void serialize(torch::serialize::OutputArchive& archive) const override
    {
        archive.write("lr", this->lr);
        archive.write("beta", beta());
        archive.write("eps", eps());
        archive.write("weight_decay", weight_decay());
    }

    void deserialize(torch::serialize::InputArchive& archive)
    {
        c10::IValue ivalue;
        if (archive.try_read("lr", ivalue)) { this->lr=ivalue.toDouble(); }
        else { TORCH_WARN("Could not read 'lr' for DiagonalAdaFisherOptions"); }

        if (archive.try_read("beta", ivalue)) { beta_ = ivalue.toDouble(); }
        else { TORCH_WARN("Could not read 'beta' for DiagonalAdaFisherOptions"); }

        if (archive.try_read("eps", ivalue)) { eps_ = ivalue.toDouble(); }
        else { TORCH_WARN("Could not read 'eps' for DiagonalAdaFisherOptions"); }

        if (archive.try_read("weight_decay", ivalue)) { weight_decay_ = ivalue.toDouble(); }
        else { TORCH_WARN("Could not read 'weight_decay' for DiagonalAdaFisherOptions"); }
    }

    std::unique_ptr<torch::optim::OptimizerOptions> clone() const override
    {
        auto cloned_options = std::make_unique<DiagonalAdaFisherOptions>(this->lr);
        cloned_options->beta(this->beta());
        cloned_options->eps(this->eps());
        cloned_options->weight_decay(this->weight_decay());
        return cloned_options;
    }
};

// --- Parameter State for DiagonalAdaFisher ---
struct DiagonalAdaFisherParamState : torch::optim::OptimizerParamState
{
    TORCH_ARG(torch::Tensor, step); // Optional, but good for potential bias correction if added
    TORCH_ARG(torch::Tensor, fisher_diag_ema); // EMA of squared gradients (g_t^2)

    // DiagonalAdaFisherParamState() = default;

    void serialize(torch::serialize::OutputArchive& archive) const override
    {
        archive.write("step", step(), /*is_buffer=*/true);
        archive.write("fisher_diag_ema", fisher_diag_ema(), /*is_buffer=*/true);
    }

    void deserialize(torch::serialize::InputArchive& archive)
    {
        at::Tensor temp_tensor;
        if (archive.try_read("step", temp_tensor, true)) { step_ = temp_tensor; }
        else
        {
            TORCH_WARN("Could not read 'step' for DiagonalAdaFisherParamState");
            if (!step_.defined()) step_ = torch::empty({0});
        }

        if (archive.try_read("fisher_diag_ema", temp_tensor, true)) { fisher_diag_ema_ = temp_tensor; }
        else
        {
            TORCH_WARN("Could not read 'fisher_diag_ema' for DiagonalAdaFisherParamState");
            if (!fisher_diag_ema_.defined()) fisher_diag_ema_ = torch::empty({0});
        }
    }

    std::unique_ptr<OptimizerParamState> clone() const override
    {
        auto cloned = std::make_unique<DiagonalAdaFisherParamState>();
        if (step_.defined()) cloned->step(step_.clone());
        if (fisher_diag_ema_.defined()) cloned->fisher_diag_ema(fisher_diag_ema_.clone());
        return cloned;
    }
};

// --- DiagonalAdaFisher Optimizer Class ---
class DiagonalAdaFisher : public torch::optim::Optimizer
{
public:
    DiagonalAdaFisher(std::vector<torch::Tensor> params, DiagonalAdaFisherOptions options);
    explicit DiagonalAdaFisher(std::vector<torch::Tensor> params, double lr = 1e-2);

    using LossClosure = std::function<torch::Tensor()>;
    torch::Tensor step(LossClosure closure = nullptr) override;
    void save(torch::serialize::OutputArchive& archive) const override;
    void load(torch::serialize::InputArchive& archive) override;

protected:
    std::unique_ptr<torch::optim::OptimizerParamState> make_param_state();
};

#endif // DIAG_ADA_FISHER_HPP
