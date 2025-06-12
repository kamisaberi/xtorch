#include "include/optimizations/hgs.h"
#include <stdexcept>

// --- HGSOptions Methods ---
void HGSOptions::serialize(torch::serialize::OutputArchive& archive) const {
    archive.write("lr", this->lr());
    archive.write("beta1", beta1());
    archive.write("beta2", beta2());
    archive.write("eps", eps());
    archive.write("weight_decay", weight_decay());
    archive.write("compression_ratio", compression_ratio());
    archive.write("sampling_beta", sampling_beta());
}

void HGSOptions::deserialize(torch::serialize::InputArchive& archive) {
    c10::IValue ivalue;
    if (archive.try_read("lr", ivalue)) { this->lr(ivalue.toDouble()); }
    if (archive.try_read("beta1", ivalue)) { beta1_ = ivalue.toDouble(); }
    if (archive.try_read("beta2", ivalue)) { beta2_ = ivalue.toDouble(); }
    if (archive.try_read("eps", ivalue)) { eps_ = ivalue.toDouble(); }
    if (archive.try_read("weight_decay", ivalue)) { weight_decay_ = ivalue.toDouble(); }
    if (archive.try_read("compression_ratio", ivalue)) { compression_ratio_ = ivalue.toDouble(); }
    if (archive.try_read("sampling_beta", ivalue)) { sampling_beta_ = ivalue.toDouble(); }
}

std::unique_ptr<torch::optim::OptimizerOptions> HGSOptions::clone() const {
    auto cloned = std::make_unique<HGSOptions>(this->lr());
    cloned->beta1(beta1()).beta2(beta2()).eps(eps()).weight_decay(weight_decay())
          .compression_ratio(compression_ratio()).sampling_beta(sampling_beta());
    return cloned;
}

// --- HGSParamState Methods ---
void HGSParamState::serialize(torch::serialize::OutputArchive& archive) const {
    archive.write("step", step(), true);
    if(error_feedback().defined()) archive.write("error_feedback", error_feedback(), true);
    if(sampling_scores().defined()) archive.write("sampling_scores", sampling_scores(), true);
    if(exp_avg().defined()) archive.write("exp_avg", exp_avg(), true);
    if(exp_avg_sq().defined()) archive.write("exp_avg_sq", exp_avg_sq(), true);
}

void HGSParamState::deserialize(torch::serialize::InputArchive& archive) {
    at::Tensor temp;
    if(archive.try_read("step", temp, true)) { step_ = temp; }
    if(archive.try_read("error_feedback", temp, true)) { error_feedback_ = temp; }
    if(archive.try_read("sampling_scores", temp, true)) { sampling_scores_ = temp; }
    if(archive.try_read("exp_avg", temp, true)) { exp_avg_ = temp; }
    if(archive.try_read("exp_avg_sq", temp, true)) { exp_avg_sq_ = temp; }
}

std::unique_ptr<torch::optim::OptimizerParamState> HGSParamState::clone() const {
    auto cloned = std::make_unique<HGSParamState>();
    if(step().defined()) cloned->step(step().clone());
    if(error_feedback().defined()) cloned->error_feedback(error_feedback().clone());
    if(sampling_scores().defined()) cloned->sampling_scores(sampling_scores().clone());
    if(exp_avg().defined()) cloned->exp_avg(exp_avg().clone());
    if(exp_avg_sq().defined()) cloned->exp_avg_sq(exp_avg_sq().clone());
    return cloned;
}

// --- HGS Implementation ---
HGS::HGS(std::vector<torch::Tensor> params, HGSOptions options)
    : torch::optim::Optimizer(std::move(params), std::make_unique<HGSOptions>(options)) {}

torch::Tensor HGS::step(LossClosure closure) {
    torch::NoGradGuard no_grad;
    torch::Tensor loss = {};
    if (closure) { loss = closure(); }

    auto& group_options = static_cast<HGSOptions&>(param_groups_[0].options());

    for (auto& p : param_groups_[0].params()) {
        if (!p.grad().defined()) { continue; }

        auto grad = p.grad();
        auto& state = static_cast<HGSParamState&>(*state_.at(p.unsafeGetTensorImpl()));

        if (!state.step().defined()) {
            state.step(torch::tensor(0.0, torch::dtype(torch::kFloat64).device(torch::kCPU)));
            state.error_feedback(torch::zeros_like(p));
            state.sampling_scores(torch::zeros_like(p));
            state.exp_avg(torch::zeros_like(p));
            state.exp_avg_sq(torch::zeros_like(p));
        }
        state.step(state.step() + 1.0);
        double current_step_val = state.step().item<double>();

        // 1. Compensate gradient with local error feedback
        auto compensated_grad = grad + state.error_feedback();

        // 2. Update adaptive sampling scores (the "Hybrid" part)
        // The score is an EMA of |g| * sqrt(v_t), combining magnitude and curvature.
        auto& v = state.exp_avg_sq(); // v_t from the inner Adam
        v.mul_(group_options.beta2()).addcmul_(grad, grad, 1.0 - group_options.beta2());

        // Use the bias-corrected v_t for a more accurate curvature estimate
        double bias_correction2 = 1.0 - std::pow(group_options.beta2(), current_step_val);
        auto v_hat = v / bias_correction2;

        // The importance score for this step
        auto current_importance = compensated_grad.abs() * v_hat.sqrt();

        // Update the long-term sampling score EMA
        auto& sampling_scores = state.sampling_scores();
        sampling_scores.mul_(group_options.sampling_beta()).add_(current_importance, 1.0 - group_options.sampling_beta());

        // 3. Select Top-K gradient values based on the scores
        int64_t k = std::max(1L, static_cast<int64_t>(p.numel() * group_options.compression_ratio()));
        auto threshold = std::get<0>(torch::kthvalue(sampling_scores.flatten(), sampling_scores.numel() - k));

        auto mask = (sampling_scores >= threshold).to(grad.dtype());
        auto sparse_grad = compensated_grad * mask;

        // 4. Update local error feedback buffer
        state.error_feedback(compensated_grad - sparse_grad);

        // 5. Apply the base Adam optimizer using the sparse gradient
        if (group_options.weight_decay() > 0.0) {
            // Note: Weight decay is applied to the parameter, not the sparse gradient,
            // for decoupled AdamW-style decay, which is more robust.
            p.data().mul_(1.0 - group_options.lr() * group_options.weight_decay());
        }

        auto& m = state.exp_avg();
        // Update momentum with the sparse gradient
        m.mul_(group_options.beta1()).add_(sparse_grad, 1.0 - group_options.beta1());

        double bias_correction1 = 1.0 - std::pow(group_options.beta1(), current_step_val);
        auto m_hat = m / bias_correction1;

        // The v_t state was already updated in step 2. We just need to use it.
        // We use the same v_hat for the denominator.
        auto denom = v_hat.sqrt().add_(group_options.eps());

        p.data().addcdiv_(m_hat, denom, -group_options.lr());
    }
    return loss;
}

// --- Boilerplate Methods ---
void HGS::save(torch::serialize::OutputArchive& archive) const { torch::optim::Optimizer::save(archive); }
void HGS::load(torch::serialize::InputArchive& archive) { torch::optim::Optimizer::load(archive); }
std::unique_ptr<torch::optim::OptimizerParamState> HGS::make_param_state() { return std::make_unique<HGSParamState>(); }