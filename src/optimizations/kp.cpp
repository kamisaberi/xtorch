#include "include/optimizations/kp.h"
#include <stdexcept>

namespace xt::optim
{
    // --- KPOptions Methods ---
    void KPOptions::serialize(torch::serialize::OutputArchive& archive) const {
        archive.write("lr", this->lr());
        archive.write("beta1", beta1()); archive.write("beta2", beta2());
        archive.write("damping", damping()); archive.write("weight_decay", weight_decay());
        archive.write("root_order", static_cast<int64_t>(root_order()));
        archive.write("precondition_frequency", precondition_frequency());
        archive.write("fallback_beta1", fallback_beta1()); archive.write("fallback_beta2", fallback_beta2());
        archive.write("fallback_eps", fallback_eps());
    }
    void KPOptions::deserialize(torch::serialize::InputArchive& archive) {
        c10::IValue ivalue;
        if (archive.try_read("lr", ivalue)) { this->lr(ivalue.toDouble()); }
        if (archive.try_read("beta1", ivalue)) { beta1_ = ivalue.toDouble(); }
        if (archive.try_read("beta2", ivalue)) { beta2_ = ivalue.toDouble(); }
        if (archive.try_read("damping", ivalue)) { damping_ = ivalue.toDouble(); }
        if (archive.try_read("weight_decay", ivalue)) { weight_decay_ = ivalue.toDouble(); }
        if (archive.try_read("root_order", ivalue)) { root_order_ = ivalue.toInt(); }
        if (archive.try_read("precondition_frequency", ivalue)) { precondition_frequency_ = ivalue.toInt(); }
        if (archive.try_read("fallback_beta1", ivalue)) { fallback_beta1_ = ivalue.toDouble(); }
        if (archive.try_read("fallback_beta2", ivalue)) { fallback_beta2_ = ivalue.toDouble(); }
        if (archive.try_read("fallback_eps", ivalue)) { fallback_eps_ = ivalue.toDouble(); }
    }
    std::unique_ptr<torch::optim::OptimizerOptions> KPOptions::clone() const {
        auto cloned = std::make_unique<KPOptions>(this->lr());
        cloned->beta1(beta1()).beta2(beta2()).damping(damping()).weight_decay(weight_decay())
              .root_order(root_order()).precondition_frequency(precondition_frequency())
              .fallback_beta1(fallback_beta1()).fallback_beta2(fallback_beta2()).fallback_eps(fallback_eps());
        return cloned;
    }

    // --- KPParamState Methods ---
    void KPParamState::serialize(torch::serialize::OutputArchive& archive) const {
        archive.write("step", step(), true);
        if(momentum_buffer().defined()) archive.write("momentum_buffer", momentum_buffer(), true);
        if(l_ema.defined()) archive.write("l_ema", l_ema, true);
        if(r_ema.defined()) archive.write("r_ema", r_ema, true);
        if(l_inv_root.defined()) archive.write("l_inv_root", l_inv_root, true);
        if(r_inv_root.defined()) archive.write("r_inv_root", r_inv_root, true);
        if(fallback_exp_avg().defined()) archive.write("fallback_exp_avg", fallback_exp_avg(), true);
        if(fallback_exp_avg_sq().defined()) archive.write("fallback_exp_avg_sq", fallback_exp_avg_sq(), true);
    }
    void KPParamState::deserialize(torch::serialize::InputArchive& archive) {
        at::Tensor temp;
        if(archive.try_read("step", temp, true)) { step_ = temp; }
        if(archive.try_read("momentum_buffer", temp, true)) { momentum_buffer_ = temp; }
        if(archive.try_read("l_ema", temp, true)) { l_ema = temp; }
        if(archive.try_read("r_ema", temp, true)) { r_ema = temp; }
        if(archive.try_read("l_inv_root", temp, true)) { l_inv_root = temp; }
        if(archive.try_read("r_inv_root", temp, true)) { r_inv_root = temp; }
        if(archive.try_read("fallback_exp_avg", temp, true)) { fallback_exp_avg_ = temp; }
        if(archive.try_read("fallback_exp_avg_sq", temp, true)) { fallback_exp_avg_sq_ = temp; }
    }
    std::unique_ptr<torch::optim::OptimizerParamState> KPParamState::clone() const {
        auto cloned = std::make_unique<KPParamState>();
        if(step().defined()) cloned->step(step().clone());
        if(momentum_buffer().defined()) cloned->momentum_buffer(momentum_buffer().clone());
        if(l_ema.defined()) cloned->l_ema = l_ema.clone();
        if(r_ema.defined()) cloned->r_ema = r_ema.clone();
        if(l_inv_root.defined()) cloned->l_inv_root = l_inv_root.clone();
        if(r_inv_root.defined()) cloned->r_inv_root = r_inv_root.clone();
        if(fallback_exp_avg().defined()) cloned->fallback_exp_avg(fallback_exp_avg().clone());
        if(fallback_exp_avg_sq().defined()) cloned->fallback_exp_avg_sq(fallback_exp_avg_sq().clone());
        return cloned;
    }


    // --- KP Optimizer Implementation ---
    KPOptimizer::KPOptimizer(std::vector<torch::Tensor> params, KPOptions options)
        : torch::optim::Optimizer(std::move(params), std::make_unique<KPOptions>(options)) {}

    torch::Tensor KPOptimizer::step(LossClosure closure) {
        torch::NoGradGuard no_grad;
        torch::Tensor loss = {};
        if (closure) { loss = closure(); }

        auto& group_options = static_cast<KPOptions&>(param_groups_[0].options());

        for (auto& p : param_groups_[0].params()) {
            if (!p.grad().defined()) { continue; }

            auto grad = p.grad();
            auto& state = static_cast<KPParamState&>(*state_.at(p.unsafeGetTensorImpl()));

            if (!state.step().defined()) {
                state.step(torch::tensor(0.0, torch::dtype(torch::kFloat64).device(torch::kCPU)));
                state.momentum_buffer(torch::zeros_like(p));
                state.fallback_exp_avg(torch::zeros_like(p));
                state.fallback_exp_avg_sq(torch::zeros_like(p));
            }
            state.step(state.step() + 1.0);
            double current_step_val = state.step().item<double>();

            if (p.dim() <= 1) {
                _fallback_to_adam(p, grad, state, group_options);
                continue;
            }

            // --- Main KP Logic ---
            torch::Tensor grad_reshaped = (p.dim() == 2) ? grad : grad.reshape({p.size(0), -1});

            // 1. Update Factor Statistics (EMAs)
            auto l_stat = grad_reshaped.t().matmul(grad_reshaped);
            auto r_stat = grad_reshaped.matmul(grad_reshaped.t());

            if (!state.l_ema.defined()) {
                state.l_ema = torch::zeros_like(l_stat);
                state.r_ema = torch::zeros_like(r_stat);
                state.l_inv_root = torch::eye(l_stat.size(0), l_stat.options());
                state.r_inv_root = torch::eye(r_stat.size(0), r_stat.options());
            }
            state.l_ema.mul_(group_options.beta2()).add_(l_stat, 1.0 - group_options.beta2());
            state.r_ema.mul_(group_options.beta2()).add_(r_stat, 1.0 - group_options.beta2());

            // 2. Update Inverse Roots (infrequently)
            if (static_cast<long>(current_step_val) % group_options.precondition_frequency() == 0) {
                state.l_inv_root = _compute_matrix_inverse_root(state.l_ema, group_options.damping(), group_options.root_order());
                state.r_inv_root = _compute_matrix_inverse_root(state.r_ema, group_options.damping(), group_options.root_order());
            }

            // 3. Precondition the Gradient
            auto preconditioned_grad_reshaped = state.r_inv_root.matmul(grad_reshaped).matmul(state.l_inv_root);
            auto preconditioned_grad = (p.dim() == 2) ? preconditioned_grad_reshaped : preconditioned_grad_reshaped.reshape(p.sizes());

            // 4. Project the original gradient onto the preconditioned gradient
            // Projection scale: (dot(grad, pre_grad) / ||pre_grad||^2)
            auto dot_product = (grad * preconditioned_grad).sum();
            auto pre_grad_norm_sq = (preconditioned_grad * preconditioned_grad).sum();

            torch::Tensor projected_update;
            if (pre_grad_norm_sq.item<double>() > 1e-12) {
                auto projection_scale = dot_product / pre_grad_norm_sq;
                projected_update = projection_scale * preconditioned_grad;
            } else {
                // If preconditioned grad is zero, there's no direction to project onto.
                projected_update = torch::zeros_like(p);
            }

            // 5. Apply Momentum to the projected update
            auto& momentum_buffer = state.momentum_buffer();
            momentum_buffer.mul_(group_options.beta1()).add_(projected_update, 1.0 - group_options.beta1());

            // 6. Apply Decoupled Weight Decay & Final Update
            if (group_options.weight_decay() > 0.0) {
                p.data().add_(p.data(), -group_options.lr() * group_options.weight_decay());
            }
            p.data().add_(momentum_buffer, -group_options.lr());
        }
        return loss;
    }

    // --- Helper Functions ---
    void KPOptimizer::_fallback_to_adam(torch::Tensor& param, const torch::Tensor& grad, KPParamState& state, const KPOptions& options) {
        auto& m = state.fallback_exp_avg();
        auto& v = state.fallback_exp_avg_sq();
        double bias_correction1 = 1.0 - std::pow(options.fallback_beta1(), state.step().item<double>());
        double bias_correction2 = 1.0 - std::pow(options.fallback_beta2(), state.step().item<double>());
        m.mul_(options.fallback_beta1()).add_(grad, 1.0 - options.fallback_beta1());
        v.mul_(options.fallback_beta2()).addcmul_(grad, grad, 1.0 - options.fallback_beta2());
        auto m_hat = m / bias_correction1;
        auto v_hat = v / bias_correction2;
        auto denom = v_hat.sqrt().add_(options.fallback_eps());
        if (options.weight_decay() > 0.0) {
            param.data().add_(param.data(), -options.lr() * options.weight_decay());
        }
        param.data().addcdiv_(m_hat, denom, -options.lr());
    }

    torch::Tensor KPOptimizer::_compute_matrix_inverse_root(const torch::Tensor& matrix, double damping, int root_order) {
        auto damped_matrix = matrix + torch::eye(matrix.size(0), matrix.options()) * damping;
        std::tuple<torch::Tensor, torch::Tensor> eigh_result;
        try {
            eigh_result = xt::linalg::eigh(damped_matrix, "U");
        } catch (const c10::Error& e) {
            return torch::eye(matrix.size(0), matrix.options());
        }
        auto eigenvalues = std::get<0>(eigh_result);
        auto eigenvectors = std::get<1>(eigh_result);
        torch::Tensor inv_root_eigenvalues = eigenvalues.clamp_min(0).pow(-1.0 / static_cast<double>(root_order));
        return eigenvectors.matmul(torch::diag(inv_root_eigenvalues).matmul(eigenvectors.t()));
    }

    // --- Boilerplate ---
    void KPOptimizer::save(torch::serialize::OutputArchive& archive) const { torch::optim::Optimizer::save(archive); }
    void KPOptimizer::load(torch::serialize::InputArchive& archive) { torch::optim::Optimizer::load(archive); }
    std::unique_ptr<torch::optim::OptimizerParamState> KPOptimizer::make_param_state() { return std::make_unique<KPParamState>(); }
}