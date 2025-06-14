#include "include/optimizations/demon.h"
#include <stdexcept>
#include <algorithm> // For std::min

// --- DemonOptions Methods ---
void DemonOptions::serialize(torch::serialize::OutputArchive& archive) const {
    archive.write("lr", this->lr());
    archive.write("beta_initial", beta_initial());
    archive.write("beta_final", beta_final());
    archive.write("total_steps", total_steps());
    archive.write("weight_decay", weight_decay());
}
void DemonOptions::deserialize(torch::serialize::InputArchive& archive) {
    c10::IValue ivalue;
    if (archive.try_read("lr", ivalue)) { this->lr(ivalue.toDouble()); }
    if (archive.try_read("beta_initial", ivalue)) { beta_initial_ = ivalue.toDouble(); }
    if (archive.try_read("beta_final", ivalue)) { beta_final_ = ivalue.toDouble(); }
    if (archive.try_read("total_steps", ivalue)) { total_steps_ = ivalue.toInt(); }
    if (archive.try_read("weight_decay", ivalue)) { weight_decay_ = ivalue.toDouble(); }
}
std::unique_ptr<torch::optim::OptimizerOptions> DemonOptions::clone() const {
    auto cloned = std::make_unique<DemonOptions>(this->lr());
    cloned->beta_initial(beta_initial()).beta_final(beta_final())
          .total_steps(total_steps()).weight_decay(weight_decay());
    return cloned;
}

// --- DemonParamState Methods ---
void DemonParamState::serialize(torch::serialize::OutputArchive& archive) const {
    archive.write("step", step(), true);
    if(momentum_buffer().defined()) archive.write("momentum_buffer", momentum_buffer(), true);
}
void DemonParamState::deserialize(torch::serialize::InputArchive& archive) {
    at::Tensor temp;
    if(archive.try_read("step", temp, true)) { step_ = temp; }
    if(archive.try_read("momentum_buffer", temp, true)) { momentum_buffer_ = temp; }
}
std::unique_ptr<torch::optim::OptimizerParamState> DemonParamState::clone() const {
    auto cloned = std::make_unique<DemonParamState>();
    if(step().defined()) cloned->step(step().clone());
    if(momentum_buffer().defined()) cloned->momentum_buffer(momentum_buffer().clone());
    return cloned;
}

// --- Demon Implementation ---
Demon::Demon(std::vector<torch::Tensor> params, DemonOptions options)
    : torch::optim::Optimizer(std::move(params), std::make_unique<DemonOptions>(options)) {}

Demon::Demon(std::vector<torch::Tensor> params, double lr)
    : Demon(std::move(params), DemonOptions(lr)) {}

torch::Tensor Demon::step(LossClosure closure) {
    torch::NoGradGuard no_grad;
    torch::Tensor loss = {};
    if (closure) { loss = closure(); }

    auto& group_options = static_cast<DemonOptions&>(param_groups_[0].options());

    // We need a single global step for the decay schedule
    long global_step = 0;
    if (!param_groups_.empty() && !param_groups_[0].params().empty()) {
        auto& p_ref = param_groups_[0].params()[0];
        auto& state_ref = static_cast<DemonParamState&>(*state_.at(p_ref.unsafeGetTensorImpl()));
        if (state_ref.step().defined()) {
            global_step = static_cast<long>(state_ref.step().item<double>());
        }
    }
    global_step++; // Increment for current step

    // 1. Calculate the current momentum beta for this step
    double decay_progress = std::min(1.0, static_cast<double>(global_step) / group_options.total_steps());
    double beta_t = group_options.beta_initial() -
                    (group_options.beta_initial() - group_options.beta_final()) * decay_progress;

    for (auto& group : param_groups_) {
        for (auto& p : group.params()) {
            if (!p.grad().defined()) { continue; }

            auto grad = p.grad();
            if (grad.is_sparse()) {
                throw std::runtime_error("Demon optimizer does not support sparse gradients.");
            }

            // Apply classic L2 regularization
            if (group_options.weight_decay() > 0.0) {
                grad = grad.add(p.detach(), group_options.weight_decay());
            }

            auto& state = static_cast<DemonParamState&>(*state_.at(p.unsafeGetTensorImpl()));

            if (!state.step().defined()) {
                state.step(torch::tensor(0.0, torch::dtype(torch::kFloat64).device(torch::kCPU)));
                state.momentum_buffer(torch::zeros_like(p));
            }
            state.step(state.step() + 1.0);

            // 2. Update momentum buffer using the dynamic beta_t
            auto& momentum_buffer = state.momentum_buffer();
            momentum_buffer.mul_(beta_t).add_(grad);

            // 3. Final SGD-like update
            p.data().add_(momentum_buffer, -group_options.lr());
        }
    }
    return loss;
}

// --- Boilerplate Methods ---
void Demon::save(torch::serialize::OutputArchive& archive) const { torch::optim::Optimizer::save(archive); }
void Demon::load(torch::serialize::InputArchive& archive) { torch::optim::Optimizer::load(archive); }
std::unique_ptr<torch::optim::OptimizerParamState> Demon::make_param_state() { return std::make_unique<DemonParamState>(); }