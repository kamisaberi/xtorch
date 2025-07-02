#include "include/optimizations/fa.h"
#include <stdexcept>
namespace xt::optim
{
    // --- FAOptions Methods ---
    void FAOptions::serialize(torch::serialize::OutputArchive& archive) const {
        archive.write("lr", this->lr);
        archive.write("beta1", beta1());
        archive.write("beta2", beta2());
        archive.write("eps", eps());
        archive.write("damping", damping());
        archive.write("weight_decay", weight_decay());
        archive.write("root_order", static_cast<int64_t>(root_order()));
        archive.write("update_frequency", update_frequency());
    }

    void FAOptions::deserialize(torch::serialize::InputArchive& archive) {
        c10::IValue ivalue;
        if (archive.try_read("lr", ivalue)) { this->lr=ivalue.toDouble(); }
        if (archive.try_read("beta1", ivalue)) { beta1_ = ivalue.toDouble(); }
        if (archive.try_read("beta2", ivalue)) { beta2_ = ivalue.toDouble(); }
        if (archive.try_read("eps", ivalue)) { eps_ = ivalue.toDouble(); }
        if (archive.try_read("damping", ivalue)) { damping_ = ivalue.toDouble(); }
        if (archive.try_read("weight_decay", ivalue)) { weight_decay_ = ivalue.toDouble(); }
        if (archive.try_read("root_order", ivalue)) { root_order_ = ivalue.toInt(); }
        if (archive.try_read("update_frequency", ivalue)) { update_frequency_ = ivalue.toInt(); }
    }

    std::unique_ptr<torch::optim::OptimizerOptions> FAOptions::clone() const {
        auto cloned = std::make_unique<FAOptions>(this->lr);
        cloned->beta1(beta1()).beta2(beta2()).eps(eps()).damping(damping())
              .weight_decay(weight_decay()).root_order(root_order())
              .update_frequency(update_frequency());
        return cloned;
    }

    // --- FAParamState Methods ---
    void FAParamState::serialize(torch::serialize::OutputArchive& archive) const {
        archive.write("step", step(), true);
        if(momentum().defined()) archive.write("momentum", momentum(), true);
        if(p_ema.defined()) archive.write("p_ema", p_ema, true);
        if(q_ema.defined()) archive.write("q_ema", q_ema, true);
        if(p_inv_root.defined()) archive.write("p_inv_root", p_inv_root, true);
        if(q_inv_root.defined()) archive.write("q_inv_root", q_inv_root, true);
        if(exp_avg_sq().defined()) archive.write("exp_avg_sq", exp_avg_sq(), true);
    }

    void FAParamState::deserialize(torch::serialize::InputArchive& archive) {
        at::Tensor temp;
        if(archive.try_read("step", temp, true)) { step_ = temp; }
        if(archive.try_read("momentum", temp, true)) { momentum_ = temp; }
        if(archive.try_read("p_ema", temp, true)) { p_ema = temp; }
        if(archive.try_read("q_ema", temp, true)) { q_ema = temp; }
        if(archive.try_read("p_inv_root", temp, true)) { p_inv_root = temp; }
        if(archive.try_read("q_inv_root", temp, true)) { q_inv_root = temp; }
        if(archive.try_read("exp_avg_sq", temp, true)) { exp_avg_sq_ = temp; }
    }

    std::unique_ptr<torch::optim::OptimizerParamState> FAParamState::clone() const {
        auto cloned = std::make_unique<FAParamState>();
        if(step().defined()) cloned->step(step().clone());
        if(momentum().defined()) cloned->momentum(momentum().clone());
        if(p_ema.defined()) cloned->p_ema = p_ema.clone();
        if(q_ema.defined()) cloned->q_ema = q_ema.clone();
        if(p_inv_root.defined()) cloned->p_inv_root = p_inv_root.clone();
        if(q_inv_root.defined()) cloned->q_inv_root = q_inv_root.clone();
        if(exp_avg_sq().defined()) cloned->exp_avg_sq(exp_avg_sq().clone());
        return cloned;
    }


    // --- FAOptimizer Implementation ---
    FAOptimizer::FAOptimizer(std::vector<torch::Tensor> params, FAOptions options)
        : torch::optim::Optimizer(std::move(params), std::make_unique<FAOptions>(options)) {}

    torch::Tensor FAOptimizer::step(LossClosure closure) {
        torch::NoGradGuard no_grad;
        torch::Tensor loss = {};
        if (closure) { loss = closure(); }

        auto& group_options = static_cast<FAOptions&>(param_groups_[0].options());

        for (auto& p : param_groups_[0].params()) {
            if (!p.grad().defined()) { continue; }

            auto grad = p.grad();
            auto& state = static_cast<FAParamState&>(*state_.at(p.unsafeGetTensorImpl()));

            if (!state.step().defined()) {
                state.step(torch::tensor(0.0, torch::dtype(torch::kFloat64).device(torch::kCPU)));
                state.momentum(torch::zeros_like(p));
                state.exp_avg_sq(torch::zeros_like(p)); // For fallback
            }
            state.step(state.step() + 1.0);
            double current_step_val = state.step().item<double>();

            // Fallback for 1D tensors (biases, etc.)
            if (p.dim() <= 1) {
                _fallback_to_adam(p, grad, state, group_options);
                continue;
            }

            // --- Main FA Logic ---
            torch::Tensor grad_reshaped = (p.dim() == 2) ? grad : grad.reshape({p.size(0), -1});
            int m = grad_reshaped.size(0);
            int n = grad_reshaped.size(1);

            // 1. Update Factor Statistics (EMAs)
            auto p_stat = grad_reshaped.matmul(grad_reshaped.t()) / n; // Normalize by size
            auto q_stat = grad_reshaped.t().matmul(grad_reshaped) / m; // Normalize by size

            if (!state.p_ema.defined()) {
                state.p_ema = torch::zeros_like(p_stat);
                state.q_ema = torch::zeros_like(q_stat);
                state.p_inv_root = torch::eye(m, p_stat.options());
                state.q_inv_root = torch::eye(n, q_stat.options());
            }

            state.p_ema.mul_(group_options.beta2()).add_(p_stat, 1.0 - group_options.beta2());
            state.q_ema.mul_(group_options.beta2()).add_(q_stat, 1.0 - group_options.beta2());

            // 2. Update Inverse Roots (infrequently)
            if (static_cast<long>(current_step_val) % group_options.update_frequency() == 0) {
                state.p_inv_root = _compute_matrix_inverse_root(state.p_ema, group_options.damping(), group_options.root_order());
                state.q_inv_root = _compute_matrix_inverse_root(state.q_ema, group_options.damping(), group_options.root_order());
            }

            // 3. Precondition the Gradient: G_pre = P_inv_root * G * Q_inv_root
            auto& P_inv_root = state.p_inv_root;
            auto& Q_inv_root = state.q_inv_root;

            auto preconditioned_grad_reshaped = P_inv_root.matmul(grad_reshaped).matmul(Q_inv_root);
            auto preconditioned_grad = (p.dim() == 2) ? preconditioned_grad_reshaped : preconditioned_grad_reshaped.reshape(p.sizes());

            // 4. Update Momentum using the preconditioned gradient
            auto& momentum = state.momentum();
            momentum.mul_(group_options.beta1()).add_(preconditioned_grad, 1.0 - group_options.beta1());

            // 5. Apply Decoupled Weight Decay & Final Update
            if (group_options.weight_decay() > 0.0) {
                p.data().add_(p.data(), -group_options.lr * group_options.weight_decay());
            }
            p.data().add_(momentum, -group_options.lr);
        }
        return loss;
    }

    // --- Helper Functions ---
    void FAOptimizer::_fallback_to_adam(torch::Tensor& param, const torch::Tensor& grad, FAParamState& state, const FAOptions& options) {
        // This is a standard Adam update
        auto& m = state.momentum(); // Re-use momentum state as Adam's m_t
        auto& v = state.exp_avg_sq();
        double current_step_val = state.step().item<double>();

        double bias_correction1 = 1.0 - std::pow(options.beta1(), current_step_val);
        double bias_correction2 = 1.0 - std::pow(options.beta2(), current_step_val);

        m.mul_(options.beta1()).add_(grad, 1.0 - options.beta1());
        v.mul_(options.beta2()).addcmul_(grad, grad, 1.0 - options.beta2());

        auto m_hat = m / bias_correction1;
        auto v_hat = v / bias_correction2;
        auto denom = v_hat.sqrt().add_(options.eps());

        if (options.weight_decay() > 0.0) {
            param.data().add_(param.data(), -options.lr * options.weight_decay());
        }
        param.data().addcdiv_(m_hat, denom, -options.lr);
    }

    torch::Tensor FAOptimizer::_compute_matrix_inverse_root(const torch::Tensor& matrix, double damping, int root_order) {
        auto damped_matrix = matrix + torch::eye(matrix.size(0), matrix.options()) * damping;
        std::tuple<torch::Tensor, torch::Tensor> eigh_result;
        try {
            //TODO START We should create EIGH
            throw std::runtime_error("We should create EIGH");
            // eigh_result = torch::linalg::eigh(damped_matrix, "U");
            //TODO END We should create EIGH
        } catch (const c10::Error& e) {
            TORCH_WARN("eigh failed, returning identity. Error: ", e.what());
            return torch::eye(matrix.size(0), matrix.options());
        }

        auto eigenvalues = std::get<0>(eigh_result);
        auto eigenvectors = std::get<1>(eigh_result);

        torch::Tensor inv_root_eigenvalues = eigenvalues.clamp_min(0).pow(-1.0 / static_cast<double>(root_order));

        return eigenvectors.matmul(torch::diag(inv_root_eigenvalues).matmul(eigenvectors.t()));
    }

    // --- Boilerplate ---
    void FAOptimizer::save(torch::serialize::OutputArchive& archive) const { torch::optim::Optimizer::save(archive); }
    void FAOptimizer::load(torch::serialize::InputArchive& archive) { torch::optim::Optimizer::load(archive); }
    std::unique_ptr<torch::optim::OptimizerParamState> FAOptimizer::make_param_state() { return std::make_unique<FAParamState>(); }
}