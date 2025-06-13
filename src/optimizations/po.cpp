#include "include/optimizations/po.h"
#include <stdexcept>

// --- POOptions Methods ---
void POOptions::serialize(torch::serialize::OutputArchive& archive) const {
    archive.write("lr", this->lr());
    archive.write("beta", beta());
    archive.write("weight_decay", weight_decay());
    archive.write("eps", eps());
}
void POOptions::deserialize(torch::serialize::InputArchive& archive) {
    c10::IValue ivalue;
    if (archive.try_read("lr", ivalue)) { this->lr(ivalue.toDouble()); }
    if (archive.try_read("beta", ivalue)) { beta_ = ivalue.toDouble(); }
    if (archive.try_read("weight_decay", ivalue)) { weight_decay_ = ivalue.toDouble(); }
    if (archive.try_read("eps", ivalue)) { eps_ = ivalue.toDouble(); }
}
std::unique_ptr<torch::optim::OptimizerOptions> POOptions::clone() const {
    auto cloned = std::make_unique<POOptions>(this->lr());
    cloned->beta(beta()).weight_decay(weight_decay()).eps(eps());
    return cloned;
}

// --- POParamState Methods ---
void POParamState::serialize(torch::serialize::OutputArchive& archive) const {
    archive.write("step", step(), true);
    if(exp_avg().defined()) archive.write("exp_avg", exp_avg(), true);
}
void POParamState::deserialize(torch::serialize::InputArchive& archive) {
    at::Tensor temp;
    if(archive.try_read("step", temp, true)) { step_ = temp; }
    if(archive.try_read("exp_avg", temp, true)) { exp_avg_ = temp; }
}
std::unique_ptr<torch::optim::OptimizerParamState> POParamState::clone() const {
    auto cloned = std::make_unique<POParamState>();
    if(step().defined()) cloned->step(step().clone());
    if(exp_avg().defined()) cloned->exp_avg(exp_avg().clone());
    return cloned;
}

// --- PO Optimizer Implementation ---
PO::PO(std::vector<torch::Tensor> params, POOptions options)
    : torch::optim::Optimizer(std::move(params), std::make_unique<POOptions>(options)) {}

PO::PO(std::vector<torch::Tensor> params, double lr)
    : PO(std::move(params), POOptions(lr)) {}

torch::Tensor PO::step(LossClosure closure) {
    torch::NoGradGuard no_grad;
    torch::Tensor loss = {};
    if (closure) { loss = closure(); }

    auto& group_options = static_cast<POOptions&>(param_groups_[0].options());

    for (auto& group : param_groups_) {
        for (auto& p : group.params()) {
            if (!p.grad().defined()) { continue; }

            auto grad = p.grad();
            if (grad.is_sparse()) {
                throw std::runtime_error("PO optimizer does not support sparse gradients.");
            }

            auto& state = static_cast<POParamState&>(*state_.at(p.unsafeGetTensorImpl()));

            if (!state.step().defined()) {
                state.step(torch::tensor(0.0, torch::dtype(torch::kFloat64).device(torch::kCPU)));
                state.exp_avg(torch::zeros_like(p));
            }
            state.step(state.step() + 1.0);

            // Apply decoupled weight decay
            if (group_options.weight_decay() > 0.0) {
                p.data().mul_(1.0 - group_options.lr() * group_options.weight_decay());
            }

            // 1. Update momentum (historical gradient direction)
            auto& m = state.exp_avg();
            m.mul_(group_options.beta()).add_(grad, 1.0 - group_options.beta());

            // 2. Calculate the projection scale
            // Scale = dot(current_grad, momentum) / ||momentum||^2
            auto dot_product = (grad * m).sum();
            auto momentum_norm_sq = (m * m).sum();

            double projection_scale = 1.0; // Default scale if momentum is zero
            if (momentum_norm_sq.item<double>() > group_options.eps()) {
                // We use the scalar value of the projection
                projection_scale = (dot_product / momentum_norm_sq).item<double>();
            }

            // Optional: Clamp or transform the scale to prevent extreme values
            // For example, using a sigmoid to keep it between 0 and 1,
            // or just clamping. Let's use a simple clamp for robustness.
            projection_scale = std::max(0.0, std::min(2.0, projection_scale));

            // 3. The final update uses the momentum direction, scaled by the projection
            // effective_lr = global_lr * projection_scale
            p.data().add_(m, -group_options.lr() * projection_scale);
        }
    }
    return loss;
}

// --- Boilerplate Methods ---
void PO::save(torch::serialize::OutputArchive& archive) const { torch::optim::Optimizer::save(archive); }
void PO::load(torch::serialize::InputArchive& archive) { torch::optim::Optimizer::load(archive); }
std::unique_ptr<torch::optim::OptimizerParamState> PO::make_param_state() { return std::make_unique<POParamState>(); }