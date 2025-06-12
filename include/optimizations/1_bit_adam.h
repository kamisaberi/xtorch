#pragma once

#include "common.h"

namespace xt::optimizations
{

    #ifndef ONE_BIT_ADAM_HPP
#define ONE_BIT_ADAM_HPP

#include <torch/torch.h>
#include <cmath> // For std::pow, std::sqrt
#include <vector>
#include <memory> // For std::unique_ptr

// Define custom options for OneBitAdam
struct OneBitAdamOptions : torch::optim::OptimizerOptions {
    OneBitAdamOptions(double lr = 1e-3) : torch::optim::OptimizerOptions(lr) {}
    TORCH_ARG(double, beta1) = 0.9;
    TORCH_ARG(double, beta2) = 0.999;
    TORCH_ARG(double, eps) = 1e-8;
    TORCH_ARG(double, weight_decay) = 0.0;
    TORCH_ARG(long, warmup_steps) = 1000;

    // Serialization
    void serialize(torch::serialize::OutputArchive& archive) const override {
        torch::optim::OptimizerOptions::serialize(archive);
        archive.write("beta1", beta1());
        archive.write("beta2", beta2());
        archive.write("eps", eps());
        archive.write("weight_decay", weight_decay());
        archive.write("warmup_steps", warmup_steps());
    }

    void deserialize(torch::serialize::InputArchive& archive) override {
        torch::optim::OptimizerOptions::deserialize(archive);
        beta1_ = archive.read_double("beta1");
        beta2_ = archive.read_double("beta2");
        eps_ = archive.read_double("eps");
        weight_decay_ = archive.read_double("weight_decay");
        warmup_steps_ = archive.read_int64("warmup_steps");
    }

    // clone() is essential for OptimizerOptions derivatives
    // It's part of the base class and needs to be overridden if you have custom options logic
    // or if the base clone isn't sufficient (though often it is if TORCH_ARG handles members).
    // For simple structs like this, the default behavior of OptimizerOptions::clone
    // might be enough if you didn't add non-TORCH_ARG members.
    // However, it's good practice to provide an explicit clone.
    std::unique_ptr<torch::optim::OptimizerOptions> clone() const override {
        return std::make_unique<OneBitAdamOptions>(*this);
    }
};

// Define custom parameter state for OneBitAdam
struct OneBitAdamParamState : torch::optim::OptimizerParamState {
    TORCH_ARG(torch::Tensor, step);
    TORCH_ARG(torch::Tensor, exp_avg);
    TORCH_ARG(torch::Tensor, exp_avg_sq);
    TORCH_ARG(torch::Tensor, error_feedback);
    TORCH_ARG(torch::Tensor, momentum_buffer);

    OneBitAdamParamState() = default;

    // Serialization
    void serialize(torch::serialize::OutputArchive& archive) const override {
        archive.write("step", step());
        archive.write("exp_avg", exp_avg());
        archive.write("exp_avg_sq", exp_avg_sq());
        archive.write("error_feedback", error_feedback());
        archive.write("momentum_buffer", momentum_buffer());
    }

    void deserialize(torch::serialize::InputArchive& archive) override {
        step_ = archive.read_tensor("step");
        exp_avg_ = archive.read_tensor("exp_avg");
        exp_avg_sq_ = archive.read_tensor("exp_avg_sq");
        error_feedback_ = archive.read_tensor("error_feedback");
        momentum_buffer_ = archive.read_tensor("momentum_buffer");
    }

    // clone() is crucial for OptimizerParamState
    std::unique_ptr<OptimizerParamState> clone() const override {
        auto cloned = std::make_unique<OneBitAdamParamState>();
        if (step_.defined()) cloned->step(step_.clone());
        if (exp_avg_.defined()) cloned->exp_avg(exp_avg_.clone());
        if (exp_avg_sq_.defined()) cloned->exp_avg_sq(exp_avg_sq_.clone());
        if (error_feedback_.defined()) cloned->error_feedback(error_feedback_.clone());
        if (momentum_buffer_.defined()) cloned->momentum_buffer(momentum_buffer_.clone());
        return cloned;
    }
};


class OneBitAdam : public torch::optim::Optimizer {
public:
    // Constructor taking parameters and options
    OneBitAdam(std::vector<torch::Tensor> params, OneBitAdamOptions options);

    // Constructor taking parameters and default options (or specific lr)
    explicit OneBitAdam(std::vector<torch::Tensor> params, double lr = 1e-3);

    // The core step function
    using LossClosure = std::function<torch::Tensor()>;
    torch::Tensor step(LossClosure closure = nullptr) override;

    // Save optimizer state
    void save(torch::serialize::OutputArchive& archive) const override;

    // Load optimizer state
    void load(torch::serialize::InputArchive& archive) override;

protected:
    // Factory method for creating parameter states, crucial for loading.
    std::unique_ptr<torch::optim::OptimizerParamState> make_param_state() override;
};

#endif // ONE_BIT_ADAM_HPP



}
