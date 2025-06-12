
// #error "COMPILING THIS EXACT HEADER FILE NOW" // Add this as the first line

#ifndef ONE_BIT_ADAM_HPP
#define ONE_BIT_ADAM_HPP

#include <torch/torch.h>
#include <torch/serialize/archive.h>

#include <cmath>
#include <vector>
#include <memory>
#include <string>
#include <cstdint>

// Define custom options for OneBitAdam
struct OneBitAdamOptions : public torch::optim::OptimizerOptions
{
public: // Explicitly public
    double lr;

    explicit OneBitAdamOptions(double learning_rate = 1e-3)
        : torch::optim::OptimizerOptions()
    {
        this->lr = learning_rate;
    }

    TORCH_ARG(double, beta1) = 0.9;
    TORCH_ARG(double, beta2) = 0.999;
    TORCH_ARG(double, eps) = 1e-8;
    TORCH_ARG(double, weight_decay) = 0.0;
    TORCH_ARG(long, warmup_steps) = 1000;

    void serialize(torch::serialize::OutputArchive& archive) const override
    {
        archive.write("lr", this->lr);
        archive.write("beta1", beta1());
        archive.write("beta2", beta2());
        archive.write("eps", eps());
        archive.write("weight_decay", weight_decay());
        archive.write("warmup_steps", warmup_steps());
    }

    void deserialize(torch::serialize::InputArchive& archive)
    {
        c10::IValue ivalue;

        if (archive.try_read("lr", ivalue)) { this->lr = ivalue.toDouble(); }
        else { TORCH_WARN("Could not read 'lr' from archive. Optimizer 'lr' remains: ", this->lr); }

        if (archive.try_read("beta1", ivalue)) { beta1_ = ivalue.toDouble(); }
        else { TORCH_WARN("Could not read 'beta1' from archive. 'beta1' remains: ", beta1()); }

        if (archive.try_read("beta2", ivalue)) { beta2_ = ivalue.toDouble(); }
        else { TORCH_WARN("Could not read 'beta2' from archive. 'beta2' remains: ", beta2()); }

        if (archive.try_read("eps", ivalue)) { eps_ = ivalue.toDouble(); }
        else { TORCH_WARN("Could not read 'eps' from archive. 'eps' remains: ", eps()); }

        if (archive.try_read("weight_decay", ivalue)) { weight_decay_ = ivalue.toDouble(); }
        else { TORCH_WARN("Could not read 'weight_decay' from archive. 'weight_decay' remains: ", weight_decay()); }

        if (archive.try_read("warmup_steps", ivalue)) { warmup_steps_ = static_cast<long>(ivalue.toInt()); }
        else { TORCH_WARN("Could not read 'warmup_steps' from archive. 'warmup_steps' remains: ", warmup_steps()); }
    }

    std::unique_ptr<torch::optim::OptimizerOptions> clone() const override
    {
        auto cloned_options = std::make_unique<OneBitAdamOptions>(this->lr);
        cloned_options->beta1(this->beta1());
        cloned_options->beta2(this->beta2());
        cloned_options->eps(this->eps());
        cloned_options->weight_decay(this->weight_decay());
        cloned_options->warmup_steps(this->warmup_steps());
        return cloned_options;
    }
};

// --- OneBitAdamParamState ---
struct OneBitAdamParamState : public torch::optim::OptimizerParamState
{
public:
    TORCH_ARG(torch::Tensor, step);
    TORCH_ARG(torch::Tensor, exp_avg);
    TORCH_ARG(torch::Tensor, exp_avg_sq);
    TORCH_ARG(torch::Tensor, error_feedback);
    TORCH_ARG(torch::Tensor, momentum_buffer);

    // OneBitAdamParamState() = default; // Now it's explicitly public

    void serialize(torch::serialize::OutputArchive& archive) const override
    {
        archive.write("step", step(), /*is_buffer=*/true);
        archive.write("exp_avg", exp_avg(), /*is_buffer=*/true);
        archive.write("exp_avg_sq", exp_avg_sq(), /*is_buffer=*/true);
        archive.write("error_feedback", error_feedback(), /*is_buffer=*/true);
        archive.write("momentum_buffer", momentum_buffer(), /*is_buffer=*/true);
    }

    void deserialize(torch::serialize::InputArchive& archive)
    {
        at::Tensor temp_tensor;

        if (archive.try_read("step", temp_tensor, /*is_buffer=*/true)) { step_ = temp_tensor; }
        else
        {
            TORCH_WARN("Could not read 'step' tensor from archive.");
            if (!step_.defined()) step_ = torch::empty({0});
        }

        if (archive.try_read("exp_avg", temp_tensor, /*is_buffer=*/true)) { exp_avg_ = temp_tensor; }
        else
        {
            TORCH_WARN("Could not read 'exp_avg' tensor from archive.");
            if (!exp_avg_.defined()) exp_avg_ = torch::empty({0});
        }

        if (archive.try_read("exp_avg_sq", temp_tensor, /*is_buffer=*/true)) { exp_avg_sq_ = temp_tensor; }
        else
        {
            TORCH_WARN("Could not read 'exp_avg_sq' tensor from archive.");
            if (!exp_avg_sq_.defined()) exp_avg_sq_ = torch::empty({0});
        }

        if (archive.try_read("error_feedback", temp_tensor, /*is_buffer=*/true)) { error_feedback_ = temp_tensor; }
        else
        {
            TORCH_WARN("Could not read 'error_feedback' tensor from archive.");
            if (!error_feedback_.defined()) error_feedback_ = torch::empty({0});
        }

        if (archive.try_read("momentum_buffer", temp_tensor, /*is_buffer=*/true)) { momentum_buffer_ = temp_tensor; }
        else
        {
            TORCH_WARN("Could not read 'momentum_buffer' tensor from archive.");
            if (!momentum_buffer_.defined()) momentum_buffer_ = torch::empty({0});
        }
    }

    std::unique_ptr<OptimizerParamState> clone() const override
    {
        auto cloned = std::make_unique<OneBitAdamParamState>(); // This will now work
        if (step_.defined()) cloned->step(step_.clone());
        if (exp_avg_.defined()) cloned->exp_avg(exp_avg_.clone());
        if (exp_avg_sq_.defined()) cloned->exp_avg_sq(exp_avg_sq_.clone());
        if (error_feedback_.defined()) cloned->error_feedback(error_feedback_.clone());
        if (momentum_buffer_.defined()) cloned->momentum_buffer(momentum_buffer_.clone());
        return cloned;
    }
};

// --- OneBitAdam Class Definition (no change from before) ---
class OneBitAdam : public torch::optim::Optimizer
{
public:
    OneBitAdam(std::vector<torch::Tensor> params, OneBitAdamOptions options);
    explicit OneBitAdam(std::vector<torch::Tensor> params, double lr = 1e-3);

    using LossClosure = std::function<torch::Tensor()>;
    torch::Tensor step(LossClosure closure = nullptr) override;
    void save(torch::serialize::OutputArchive& archive) const override;
    void load(torch::serialize::InputArchive& archive) override;

    // protected: // Removed protected, not strictly needed for make_param_state as Optimizer base handles it.
    //     std::unique_ptr<torch::optim::OptimizerParamState> make_param_state();
};

#endif // ONE_BIT_ADAM_HPP
