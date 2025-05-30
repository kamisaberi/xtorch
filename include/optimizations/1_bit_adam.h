#pragma once
#include "common.h"

namespace xt::optimizations
{


    class OneBitAdam : public torch::optim::Optimizer {
    public:
        explicit OneBitAdam(const std::vector<torch::Tensor>& parameters, double lr = 0.01, double momentum = 0.9)
            : torch::optim::Optimizer(std::move(parameters)), lr_(lr), momentum_(momentum) {
            velocities_.resize(param_groups()[0].params().size());
            for (size_t i = 0; i < velocities_.size(); ++i) {
                velocities_[i] = torch::zeros_like(param_groups()[0].params()[i]);
            }
        }

        void step() override {
            auto& params = param_groups()[0].params();
            for (size_t i = 0; i < params.size(); ++i) {
                auto& param = params[i];
                if (!param.grad().defined()) continue;

                auto grad = param.grad();
                velocities_[i] = momentum_ * velocities_[i] + (1 - momentum_) * grad;
                param.add_(-lr_ * velocities_[i]);
            }
        }

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