#include <optimizations/power_sgd.h>
#include <stdexcept>

namespace xt::optim
{
    // --- PowerSGDOptions Methods ---
    void PowerSGDOptions::serialize(torch::serialize::OutputArchive& archive) const {
        archive.write("lr", this->lr());
        archive.write("momentum", momentum());
        archive.write("weight_decay", weight_decay());
        archive.write("power", power());
    }
    void PowerSGDOptions::deserialize(torch::serialize::InputArchive& archive) {
        c10::IValue ivalue;
        if (archive.try_read("lr", ivalue)) { this->lr(ivalue.toDouble()); }
        if (archive.try_read("momentum", ivalue)) { momentum_ = ivalue.toDouble(); }
        if (archive.try_read("weight_decay", ivalue)) { weight_decay_ = ivalue.toDouble(); }
        if (archive.try_read("power", ivalue)) { power_ = ivalue.toDouble(); }
    }
    std::unique_ptr<torch::optim::OptimizerOptions> PowerSGDOptions::clone() const {
        auto cloned = std::make_unique<PowerSGDOptions>(this->lr());
        cloned->momentum(momentum()).weight_decay(weight_decay()).power(power());
        return cloned;
    }

    // --- PowerSGDParamState Methods ---
    void PowerSGDParamState::serialize(torch::serialize::OutputArchive& archive) const {
        if(momentum_buffer().defined()) archive.write("momentum_buffer", momentum_buffer(), true);
    }
    void PowerSGDParamState::deserialize(torch::serialize::InputArchive& archive) {
        at::Tensor temp;
        if(archive.try_read("momentum_buffer", temp, true)) { momentum_buffer_ = temp; }
    }
    std::unique_ptr<torch::optim::OptimizerParamState> PowerSGDParamState::clone() const {
        auto cloned = std::make_unique<PowerSGDParamState>();
        if(momentum_buffer().defined()) cloned->momentum_buffer(momentum_buffer().clone());
        return cloned;
    }

    // --- PowerSGD Implementation ---
    PowerSGD::PowerSGD(std::vector<torch::Tensor> params, PowerSGDOptions options)
        : torch::optim::Optimizer(std::move(params), std::make_unique<PowerSGDOptions>(options)) {}

    PowerSGD::PowerSGD(std::vector<torch::Tensor> params, double lr)
        : PowerSGD(std::move(params), PowerSGDOptions(lr)) {}

    torch::Tensor PowerSGD::step(LossClosure closure) {
        torch::NoGradGuard no_grad;
        torch::Tensor loss = {};
        if (closure) { loss = closure(); }

        auto& group_options = static_cast<PowerSGDOptions&>(param_groups_[0].options());

        for (auto& group : param_groups_) {
            for (auto& p : group.params()) {
                if (!p.grad().defined()) { continue; }

                auto grad = p.grad();
                if (grad.is_sparse()) {
                    throw std::runtime_error("PowerSGD optimizer does not support sparse gradients.");
                }

                // Apply classic weight decay (add L2 penalty to the gradient)
                if (group_options.weight_decay() > 0.0) {
                    grad = grad.add(p.detach(), group_options.weight_decay());
                }

                // 1. Apply the Power transformation to the gradient
                // powered_grad = sign(grad) * |grad|^p
                auto grad_sign = torch::sign(grad);
                auto grad_abs_powered = torch::pow(torch::abs(grad), group_options.power());
                auto powered_grad = grad_sign * grad_abs_powered;

                // 2. Perform SGD with Momentum update using the powered gradient
                auto& state = static_cast<PowerSGDParamState&>(*state_.at(p.unsafeGetTensorImpl()));
                if (!state.momentum_buffer().defined()) {
                    state.momentum_buffer(torch::zeros_like(p));
                }
                auto& momentum_buffer = state.momentum_buffer();

                // Update momentum buffer: m = beta * m + powered_grad
                momentum_buffer.mul_(group_options.momentum()).add_(powered_grad);

                // 3. Apply final update
                // p = p - lr * m
                p.data().add_(momentum_buffer, -group_options.lr());
            }
        }
        return loss;
    }

    // --- Boilerplate Methods ---
    void PowerSGD::save(torch::serialize::OutputArchive& archive) const { torch::optim::Optimizer::save(archive); }
    void PowerSGD::load(torch::serialize::InputArchive& archive) { torch::optim::Optimizer::load(archive); }
    std::unique_ptr<torch::optim::OptimizerParamState> PowerSGD::make_param_state() { return std::make_unique<PowerSGDParamState>(); }
}