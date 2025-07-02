#pragma once


#include "common.h"
// --- Options for OneBitLamb ---

namespace xt::optim
{
    struct OneBitLambOptions : torch::optim::OptimizerOptions
    {
        double lr;

        explicit OneBitLambOptions(double learning_rate = 1e-3)
            : torch::optim::OptimizerOptions()
        {
            this->lr = learning_rate; // Set lr using base class setter
        }

        TORCH_ARG(double, beta1) = 0.9;
        TORCH_ARG(double, beta2) = 0.999;
        TORCH_ARG(double, eps) = 1e-6; // LAMB often uses a slightly larger eps like 1e-6
        TORCH_ARG(double, weight_decay) = 0.01; // LAMB often has a default weight decay
        TORCH_ARG(long, warmup_steps) = 1000;
        TORCH_ARG(double, trust_clip_threshold) = 10.0;
        // To clip the trust ratio if it's too large, preventing instability. 0 means no clipping.

        void serialize(torch::serialize::OutputArchive& archive) const override
        {
            archive.write("lr", this->lr);
            archive.write("beta1", beta1());
            archive.write("beta2", beta2());
            archive.write("eps", eps());
            archive.write("weight_decay", weight_decay());
            archive.write("warmup_steps", warmup_steps());
            archive.write("trust_clip_threshold", trust_clip_threshold());
        }

        void deserialize(torch::serialize::InputArchive& archive)
        {
            c10::IValue ivalue;

            if (archive.try_read("lr", ivalue)) { this->lr = ivalue.toDouble(); }
            else { TORCH_WARN("Could not read 'lr' for OneBitLambOptions"); }

            if (archive.try_read("beta1", ivalue)) { beta1_ = ivalue.toDouble(); }
            else { TORCH_WARN("Could not read 'beta1' for OneBitLambOptions"); }

            if (archive.try_read("beta2", ivalue)) { beta2_ = ivalue.toDouble(); }
            else { TORCH_WARN("Could not read 'beta2' for OneBitLambOptions"); }

            if (archive.try_read("eps", ivalue)) { eps_ = ivalue.toDouble(); }
            else { TORCH_WARN("Could not read 'eps' for OneBitLambOptions"); }

            if (archive.try_read("weight_decay", ivalue)) { weight_decay_ = ivalue.toDouble(); }
            else { TORCH_WARN("Could not read 'weight_decay' for OneBitLambOptions"); }

            if (archive.try_read("warmup_steps", ivalue)) { warmup_steps_ = static_cast<long>(ivalue.toInt()); }
            else { TORCH_WARN("Could not read 'warmup_steps' for OneBitLambOptions"); }

            if (archive.try_read("trust_clip_threshold", ivalue)) { trust_clip_threshold_ = ivalue.toDouble(); }
            else
            {
                TORCH_WARN("Could not read 'trust_clip_threshold' for OneBitLambOptions, using default: ",
                           trust_clip_threshold());
            }
        }

        std::unique_ptr<torch::optim::OptimizerOptions> clone() const override
        {
            auto cloned_options = std::make_unique<OneBitLambOptions>(this->lr);
            cloned_options->beta1(this->beta1());
            cloned_options->beta2(this->beta2());
            cloned_options->eps(this->eps());
            cloned_options->weight_decay(this->weight_decay());
            cloned_options->warmup_steps(this->warmup_steps());
            cloned_options->trust_clip_threshold(this->trust_clip_threshold());
            return cloned_options;
        }
    };

    // --- Parameter State for OneBitLamb (identical to OneBitAdam's) ---
    struct OneBitLambParamState : public torch::optim::OptimizerParamState
    {
    public:
        TORCH_ARG(torch::Tensor, step);
        TORCH_ARG(torch::Tensor, exp_avg); // m_t
        TORCH_ARG(torch::Tensor, exp_avg_sq); // v_t
        TORCH_ARG(torch::Tensor, error_feedback); // e_t
        TORCH_ARG(torch::Tensor, momentum_buffer); // Full precision m_t for compression

        // OneBitLambParamState() = default;

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
            if (archive.try_read("step", temp_tensor, true)) { step_ = temp_tensor; }
            else
            {
                TORCH_WARN("Could not read 'step' for OneBitLambParamState");
                if (!step_.defined()) step_ = torch::empty({0});
            }

            if (archive.try_read("exp_avg", temp_tensor, true)) { exp_avg_ = temp_tensor; }
            else
            {
                TORCH_WARN("Could not read 'exp_avg' for OneBitLambParamState");
                if (!exp_avg_.defined()) exp_avg_ = torch::empty({0});
            }

            if (archive.try_read("exp_avg_sq", temp_tensor, true)) { exp_avg_sq_ = temp_tensor; }
            else
            {
                TORCH_WARN("Could not read 'exp_avg_sq' for OneBitLambParamState");
                if (!exp_avg_sq_.defined()) exp_avg_sq_ = torch::empty({0});
            }

            if (archive.try_read("error_feedback", temp_tensor, true)) { error_feedback_ = temp_tensor; }
            else
            {
                TORCH_WARN("Could not read 'error_feedback' for OneBitLambParamState");
                if (!error_feedback_.defined()) error_feedback_ = torch::empty({0});
            }

            if (archive.try_read("momentum_buffer", temp_tensor, true)) { momentum_buffer_ = temp_tensor; }
            else
            {
                TORCH_WARN("Could not read 'momentum_buffer' for OneBitLambParamState");
                if (!momentum_buffer_.defined()) momentum_buffer_ = torch::empty({0});
            }
        }

        std::unique_ptr<OptimizerParamState> clone() const override
        {
            auto cloned = std::make_unique<OneBitLambParamState>();
            if (step_.defined()) cloned->step(step_.clone());
            if (exp_avg_.defined()) cloned->exp_avg(exp_avg_.clone());
            if (exp_avg_sq_.defined()) cloned->exp_avg_sq(exp_avg_sq_.clone());
            if (error_feedback_.defined()) cloned->error_feedback(error_feedback_.clone());
            if (momentum_buffer_.defined()) cloned->momentum_buffer(momentum_buffer_.clone());
            return cloned;
        }
    };

    // --- OneBitLamb Optimizer Class ---
    class OneBitLamb : public torch::optim::Optimizer
    {
    public:
        OneBitLamb(std::vector<torch::Tensor> params, OneBitLambOptions options);
        explicit OneBitLamb(std::vector<torch::Tensor> params, double lr = 1e-3);

        using LossClosure = std::function<torch::Tensor()>;
        torch::Tensor step(LossClosure closure = nullptr) override;
        void save(torch::serialize::OutputArchive& archive) const override;
        void load(torch::serialize::InputArchive& archive) override;

    protected:
        std::unique_ptr<torch::optim::OptimizerParamState> make_param_state();
    };
}