#include "include/optimizations/plo.h"
#include <stdexcept>

namespace xt::optim
{
    // --- PLOOptions Methods ---
    void PLOOptions::serialize(torch::serialize::OutputArchive& archive) const {
        archive.write("lr", this->lr()); archive.write("alpha", alpha()); archive.write("k", k());
        archive.write("beta_momentum", beta_momentum()); archive.write("beta_stats", beta_stats());
        archive.write("damping", damping()); archive.write("weight_decay", weight_decay());
        archive.write("root_order", static_cast<int64_t>(root_order()));
        archive.write("precondition_frequency", precondition_frequency());
    }
    void PLOOptions::deserialize(torch::serialize::InputArchive& archive) {
        c10::IValue ivalue;
        if (archive.try_read("lr", ivalue)) { this->lr(ivalue.toDouble()); }
        if (archive.try_read("alpha", ivalue)) { alpha_ = ivalue.toDouble(); }
        if (archive.try_read("k", ivalue)) { k_ = ivalue.toInt(); }
        if (archive.try_read("beta_momentum", ivalue)) { beta_momentum_ = ivalue.toDouble(); }
        if (archive.try_read("beta_stats", ivalue)) { beta_stats_ = ivalue.toDouble(); }
        if (archive.try_read("damping", ivalue)) { damping_ = ivalue.toDouble(); }
        if (archive.try_read("weight_decay", ivalue)) { weight_decay_ = ivalue.toDouble(); }
        if (archive.try_read("root_order", ivalue)) { root_order_ = ivalue.toInt(); }
        if (archive.try_read("precondition_frequency", ivalue)) { precondition_frequency_ = ivalue.toInt(); }
    }
    std::unique_ptr<torch::optim::OptimizerOptions> PLOOptions::clone() const {
        auto cloned = std::make_unique<PLOOptions>(this->lr());
        cloned->alpha(alpha()).k(k()).beta_momentum(beta_momentum()).beta_stats(beta_stats())
              .damping(damping()).weight_decay(weight_decay()).root_order(root_order())
              .precondition_frequency(precondition_frequency());
        return cloned;
    }

    // --- PLOParamState Methods ---
    void PLOParamState::serialize(torch::serialize::OutputArchive& archive) const {
        archive.write("step", step(), true);
        if(slow_param().defined()) archive.write("slow_param", slow_param(), true);
        if(momentum_buffer().defined()) archive.write("momentum_buffer", momentum_buffer(), true);
        if(l_ema.defined()) archive.write("l_ema", l_ema, true);
        if(r_ema.defined()) archive.write("r_ema", r_ema, true);
        if(l_inv_root.defined()) archive.write("l_inv_root", l_inv_root, true);
        if(r_inv_root.defined()) archive.write("r_inv_root", r_inv_root, true);
    }
    void PLOParamState::deserialize(torch::serialize::InputArchive& archive) {
        at::Tensor temp;
        if(archive.try_read("step", temp, true)) { step_ = temp; }
        if(archive.try_read("slow_param", temp, true)) { slow_param_ = temp; }
        if(archive.try_read("momentum_buffer", temp, true)) { momentum_buffer_ = temp; }
        if(archive.try_read("l_ema", temp, true)) { l_ema = temp; }
        if(archive.try_read("r_ema", temp, true)) { r_ema = temp; }
        if(archive.try_read("l_inv_root", temp, true)) { l_inv_root = temp; }
        if(archive.try_read("r_inv_root", temp, true)) { r_inv_root = temp; }
    }
    std::unique_ptr<torch::optim::OptimizerParamState> PLOParamState::clone() const {
        auto cloned = std::make_unique<PLOParamState>();
        if(step().defined()) cloned->step(step().clone());
        if(slow_param().defined()) cloned->slow_param(slow_param().clone());
        if(momentum_buffer().defined()) cloned->momentum_buffer(momentum_buffer().clone());
        if(l_ema.defined()) cloned->l_ema = l_ema.clone();
        if(r_ema.defined()) cloned->r_ema = r_ema.clone();
        if(l_inv_root.defined()) cloned->l_inv_root = l_inv_root.clone();
        if(r_inv_root.defined()) cloned->r_inv_root = r_inv_root.clone();
        return cloned;
    }


    // --- PLO Implementation ---
    PLO::PLO(std::vector<torch::Tensor> params, PLOOptions options)
        : torch::optim::Optimizer(std::move(params), std::make_unique<PLOOptions>(options)) {
        for (auto& group : param_groups_) {
            for (auto& p : group.params()) {
                auto& state = static_cast<PLOParamState&>(*state_.at(p.unsafeGetTensorImpl()));
                state.slow_param(p.detach().clone());
            }
        }
    }

    torch::Tensor PLO::step(LossClosure closure) {
        torch::NoGradGuard no_grad;
        torch::Tensor loss = {};
        if (closure) { loss = closure(); }

        auto& group_options = static_cast<PLOOptions&>(param_groups_[0].options());
        long global_step = 0;

        // --- Loop 1: Inner Projected Optimizer Step ---
        // This updates the "fast weights" (p.data()) for all parameters.
        for (auto& group : param_groups_) {
            for (auto& p : group.params()) {
                if (!p.grad().defined()) { continue; }

                auto grad = p.grad();
                auto& state = static_cast<PLOParamState&>(*state_.at(p.unsafeGetTensorImpl()));

                if (!state.step().defined()) {
                    state.step(torch::tensor(0.0, torch::dtype(torch::kFloat64).device(torch::kCPU)));
                    state.momentum_buffer(torch::zeros_like(p));
                }
                state.step(state.step() + 1.0);
                double current_step_val = state.step().item<double>();
                if (global_step == 0) global_step = static_cast<long>(current_step_val);

                // Fallback for 1D tensors to simple SGD w/ momentum
                if (p.dim() <= 1) {
                    _fallback_to_sgd(p, grad, state, group_options);
                    continue;
                }

                // Main Projected update logic
                torch::Tensor grad_reshaped = (p.dim() == 2) ? grad : grad.reshape({p.size(0), -1});

                // Update Factor Statistics
                auto l_stat = grad_reshaped.t().matmul(grad_reshaped);
                auto r_stat = grad_reshaped.matmul(grad_reshaped.t());
                if (!state.l_ema.defined()) {
                    state.l_ema = torch::zeros_like(l_stat); state.r_ema = torch::zeros_like(r_stat);
                    state.l_inv_root = torch::eye(l_stat.size(0), l_stat.options());
                    state.r_inv_root = torch::eye(r_stat.size(0), r_stat.options());
                }
                state.l_ema.mul_(group_options.beta_stats()).add_(l_stat, 1.0 - group_options.beta_stats());
                state.r_ema.mul_(group_options.beta_stats()).add_(r_stat, 1.0 - group_options.beta_stats());

                // Update Inverse Roots
                if (static_cast<long>(current_step_val) % group_options.precondition_frequency() == 0) {
                    state.l_inv_root = _compute_matrix_inverse_root(state.l_ema, group_options.damping(), group_options.root_order());
                    state.r_inv_root = _compute_matrix_inverse_root(state.r_ema, group_options.damping(), group_options.root_order());
                }

                // Precondition the Gradient
                auto preconditioned_grad_reshaped = state.r_inv_root.matmul(grad_reshaped).matmul(state.l_inv_root);
                auto preconditioned_grad = (p.dim() == 2) ? preconditioned_grad_reshaped : preconditioned_grad_reshaped.reshape(p.sizes());

                // Project the original gradient onto the preconditioned gradient
                auto dot_product = (grad * preconditioned_grad).sum();
                auto pre_grad_norm_sq = (preconditioned_grad * preconditioned_grad).sum();

                torch::Tensor projected_update;
                if (pre_grad_norm_sq.item<double>() > 1e-12) {
                    auto projection_scale = dot_product / pre_grad_norm_sq;
                    projected_update = projection_scale * preconditioned_grad;
                } else {
                    projected_update = torch::zeros_like(p);
                }

                // Apply Momentum to the projected update
                auto& momentum_buffer = state.momentum_buffer();
                momentum_buffer.mul_(group_options.beta_momentum()).add_(projected_update, 1.0 - group_options.beta_momentum());

                // Apply Decoupled Weight Decay & Final Update
                if (group_options.weight_decay() > 0.0) {
                    p.data().add_(p.data(), -group_options.lr() * group_options.weight_decay());
                }
                p.data().add_(momentum_buffer, -group_options.lr());
            }
        }

        // --- Loop 2: Lookahead Synchronization Step ---
        if (global_step > 0 && global_step % group_options.k() == 0) {
            for (auto& group : param_groups_) {
                for (auto& p : group.params()) {
                    auto& state = static_cast<PLOParamState&>(*state_.at(p.unsafeGetTensorImpl()));
                    auto& slow_p = state.slow_param();
                    slow_p.add_(p.detach() - slow_p, group_options.alpha());
                    p.data().copy_(slow_p);
                }
            }
        }

        return loss;
    }

    // --- Helper Functions ---
    void PLO::_fallback_to_sgd(torch::Tensor& param, const torch::Tensor& grad, PLOParamState& state, const PLOOptions& options) {
        auto grad_with_wd = grad;
        if (options.weight_decay() > 0.0) {
            grad_with_wd = grad.add(param.detach(), options.weight_decay());
        }
        auto& momentum_buffer = state.momentum_buffer();
        momentum_buffer.mul_(options.beta_momentum()).add_(grad_with_wd);
        param.data().add_(momentum_buffer, -options.lr());
    }

    torch::Tensor PLO::_compute_matrix_inverse_root(const torch::Tensor& matrix, double damping, int root_order) {
        auto damped_matrix = matrix + torch::eye(matrix.size(0), matrix.options()) * damping;
        std::tuple<torch::Tensor, torch::Tensor> eigh_result;
        try {
            //TODO START We should create eigh
            throw std::runtime_error("We should create eigh");
            // eigh_result = torch::linalg::eigh(damped_matrix, "U");
            //TODO END We should create eigh
        } catch (const c10::Error& e) { return torch::eye(matrix.size(0), matrix.options()); }
        auto eigenvalues = std::get<0>(eigh_result);
        auto eigenvectors = std::get<1>(eigh_result);
        torch::Tensor inv_root_eigenvalues = eigenvalues.clamp_min(0).pow(-1.0 / static_cast<double>(root_order));
        return eigenvectors.matmul(torch::diag(inv_root_eigenvalues).matmul(eigenvectors.t()));
    }

    // --- Boilerplate ---
    void PLO::save(torch::serialize::OutputArchive& archive) const { torch::optim::Optimizer::save(archive); }
    void PLO::load(torch::serialize::InputArchive& archive) { torch::optim::Optimizer::load(archive); }
    std::unique_ptr<torch::optim::OptimizerParamState> PLO::make_param_state() { return std::make_unique<PLOParamState>(); }
}