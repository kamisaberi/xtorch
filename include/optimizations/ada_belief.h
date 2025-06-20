#pragma once


#include "common.h"

namespace xt::optim
{
    // --- Options for AdaBelief ---
    struct AdaBeliefOptions : torch::optim::OptimizerOptions {
        double lr;
        explicit AdaBeliefOptions(double learning_rate = 1e-3)
            : torch::optim::OptimizerOptions() {
            this->lr=learning_rate;
        }

        TORCH_ARG(double, beta1) = 0.9;
        TORCH_ARG(double, beta2) = 0.999;
        TORCH_ARG(double, eps) = 1e-8; // Original paper uses 1e-16, but 1e-8 is common in Adam-like opts
        TORCH_ARG(double, weight_decay) = 0.0;
        // TORCH_ARG(bool, amsgrad) = false; // Optional, not implemented here
        // TORCH_ARG(bool, rectify) = true; // Optional, for variance rectification like RAdam

        void serialize(torch::serialize::OutputArchive& archive) const override {
            archive.write("lr", this->lr);
            archive.write("beta1", beta1());
            archive.write("beta2", beta2());
            archive.write("eps", eps());
            archive.write("weight_decay", weight_decay());
        }

        void deserialize(torch::serialize::InputArchive& archive)  {
            c10::IValue ivalue;
            if (archive.try_read("lr", ivalue)) { this->lr=ivalue.toDouble(); }
            else { TORCH_WARN("Could not read 'lr' for AdaBeliefOptions"); }

            if (archive.try_read("beta1", ivalue)) { beta1_ = ivalue.toDouble(); }
            else { TORCH_WARN("Could not read 'beta1' for AdaBeliefOptions"); }

            if (archive.try_read("beta2", ivalue)) { beta2_ = ivalue.toDouble(); }
            else { TORCH_WARN("Could not read 'beta2' for AdaBeliefOptions"); }

            if (archive.try_read("eps", ivalue)) { eps_ = ivalue.toDouble(); }
            else { TORCH_WARN("Could not read 'eps' for AdaBeliefOptions"); }

            if (archive.try_read("weight_decay", ivalue)) { weight_decay_ = ivalue.toDouble(); }
            else { TORCH_WARN("Could not read 'weight_decay' for AdaBeliefOptions"); }
        }

        std::unique_ptr<torch::optim::OptimizerOptions> clone() const override {
            auto cloned_options = std::make_unique<AdaBeliefOptions>(this->lr);
            cloned_options->beta1(this->beta1());
            cloned_options->beta2(this->beta2());
            cloned_options->eps(this->eps());
            cloned_options->weight_decay(this->weight_decay());
            return cloned_options;
        }
    };

    // --- Parameter State for AdaBelief ---
    struct AdaBeliefParamState : torch::optim::OptimizerParamState {
        TORCH_ARG(torch::Tensor, step);
        TORCH_ARG(torch::Tensor, exp_avg);      // m_t (EMA of gradients)
        TORCH_ARG(torch::Tensor, exp_avg_var);  // s_t (EMA of (g_t - m_t)^2)
        // TORCH_ARG(torch::Tensor, max_exp_avg_var); // For AMSGrad if implemented

        // AdaBeliefParamState() = default;

        void serialize(torch::serialize::OutputArchive& archive) const override {
            archive.write("step", step(), /*is_buffer=*/true);
            archive.write("exp_avg", exp_avg(), /*is_buffer=*/true);
            archive.write("exp_avg_var", exp_avg_var(), /*is_buffer=*/true);
        }

        void deserialize(torch::serialize::InputArchive& archive)  {
            at::Tensor temp_tensor;
            if (archive.try_read("step", temp_tensor, true)) { step_ = temp_tensor; }
            else { TORCH_WARN("Could not read 'step' for AdaBeliefParamState"); if(!step_.defined()) step_ = torch::empty({0});}

            if (archive.try_read("exp_avg", temp_tensor, true)) { exp_avg_ = temp_tensor; }
            else { TORCH_WARN("Could not read 'exp_avg' for AdaBeliefParamState"); if(!exp_avg_.defined()) exp_avg_ = torch::empty({0});}

            if (archive.try_read("exp_avg_var", temp_tensor, true)) { exp_avg_var_ = temp_tensor; }
            else { TORCH_WARN("Could not read 'exp_avg_var' for AdaBeliefParamState"); if(!exp_avg_var_.defined()) exp_avg_var_ = torch::empty({0});}
        }

        std::unique_ptr<OptimizerParamState> clone() const override {
            auto cloned = std::make_unique<AdaBeliefParamState>();
            if (step_.defined()) cloned->step(step_.clone());
            if (exp_avg_.defined()) cloned->exp_avg(exp_avg_.clone());
            if (exp_avg_var_.defined()) cloned->exp_avg_var(exp_avg_var_.clone());
            return cloned;
        }
    };

    // --- AdaBelief Optimizer Class ---
    class AdaBelief : public torch::optim::Optimizer {
    public:
        AdaBelief(std::vector<torch::Tensor> params, AdaBeliefOptions options);
        explicit AdaBelief(std::vector<torch::Tensor> params, double lr = 1e-3);

        using LossClosure = std::function<torch::Tensor()>;
        torch::Tensor step(LossClosure closure = nullptr) override;
        void save(torch::serialize::OutputArchive& archive) const override;
        void load(torch::serialize::InputArchive& archive) override;

    protected:
        std::unique_ptr<torch::optim::OptimizerParamState> make_param_state() ;
    };
}