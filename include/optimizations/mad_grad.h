#pragma once

#include "common.h"

namespace xt::optimizations
{
    class MADGRAD : public torch::optim::Optimizer {
    public:
        explicit MADGRAD(std::vector<torch::Tensor>&& parameters, double lr = 0.01, double momentum = 0.9);

        using LossClosure = std::function<torch::Tensor()>;
        torch::Tensor step(LossClosure closure = nullptr) override;

        // Getter and setter for learning rate
        double lr() const { return lr_; }
        void lr(double lr) { lr_ = lr; }

        // Getter and setter for momentum
        double momentum() const { return momentum_; }
        void momentum(double momentum) { momentum_ = momentum; }

    private:
        double lr_;
        double momentum_;
        std::vector<torch::Tensor> velocities_;
    };
}