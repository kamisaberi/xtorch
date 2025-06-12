#include "include/optimizations/fasfa.h"
#include <stdexcept>

// --- FASFAOptions Methods ---
void FASFAOptions::serialize(torch::serialize::OutputArchive& archive) const
{
    archive.write("lr", this->lr);
    archive.write("beta_fast", beta_fast());
    archive.write("beta_slow", beta_slow());
    archive.write("beta1_adam", beta1_adam());
    archive.write("beta2_adam", beta2_adam());
    archive.write("eps", eps());
    archive.write("weight_decay", weight_decay());
    archive.write("root_order", static_cast<int64_t>(root_order()));
    archive.write("precondition_frequency", precondition_frequency());
    archive.write("start_preconditioning_step", start_preconditioning_step());
    archive.write("grafting_type", grafting_type());
}

void FASFAOptions::deserialize(torch::serialize::InputArchive& archive)
{
    c10::IValue ivalue;
    if (archive.try_read("lr", ivalue)) { this->lr = ivalue.toDouble(); }
    if (archive.try_read("beta_fast", ivalue)) { beta_fast_ = ivalue.toDouble(); }
    if (archive.try_read("beta_slow", ivalue)) { beta_slow_ = ivalue.toDouble(); }
    if (archive.try_read("beta1_adam", ivalue)) { beta1_adam_ = ivalue.toDouble(); }
    if (archive.try_read("beta2_adam", ivalue)) { beta2_adam_ = ivalue.toDouble(); }
    if (archive.try_read("eps", ivalue)) { eps_ = ivalue.toDouble(); }
    if (archive.try_read("weight_decay", ivalue)) { weight_decay_ = ivalue.toDouble(); }
    if (archive.try_read("root_order", ivalue)) { root_order_ = ivalue.toInt(); }
    if (archive.try_read("precondition_frequency", ivalue)) { precondition_frequency_ = ivalue.toInt(); }
    if (archive.try_read("start_preconditioning_step", ivalue)) { start_preconditioning_step_ = ivalue.toInt(); }
    if (archive.try_read("grafting_type", ivalue)) { grafting_type_ = ivalue.toStringRef(); }
}

std::unique_ptr<torch::optim::OptimizerOptions> FASFAOptions::clone() const
{
    auto cloned = std::make_unique<FASFAOptions>(this->lr);
    cloned->beta_fast(beta_fast()).beta_slow(beta_slow()).beta1_adam(beta1_adam()).beta2_adam(beta2_adam())
          .eps(eps()).weight_decay(weight_decay()).root_order(root_order())
          .precondition_frequency(precondition_frequency())
          .start_preconditioning_step(start_preconditioning_step()).grafting_type(grafting_type());
    return cloned;
}

// --- FASFAParamState Methods ---
void FASFAParamState::serialize(torch::serialize::OutputArchive& archive) const
{
    archive.write("step", step(), true);
    if (exp_avg().defined()) archive.write("exp_avg", exp_avg(), true);
    if (exp_avg_sq().defined()) archive.write("exp_avg_sq", exp_avg_sq(), true);

    archive.write("num_factors", static_cast<int64_t>(fast_ema_factors.size()));
    for (size_t i = 0; i < fast_ema_factors.size(); ++i)
    {
        archive.write("fast_ema_" + std::to_string(i), fast_ema_factors[i], true);
        archive.write("slow_ema_" + std::to_string(i), slow_ema_factors[i], true);
        archive.write("inv_root_" + std::to_string(i), inv_root_factors[i], true);
    }
}

void FASFAParamState::deserialize(torch::serialize::InputArchive& archive)
{
    at::Tensor temp;
    if (archive.try_read("step", temp, true)) { step_ = temp; }
    if (archive.try_read("exp_avg", temp, true)) { exp_avg_ = temp; }
    if (archive.try_read("exp_avg_sq", temp, true)) { exp_avg_sq_ = temp; }

    c10::IValue ivalue;
    if (archive.try_read("num_factors", ivalue))
    {
        int64_t num_factors = ivalue.toInt();
        fast_ema_factors.resize(num_factors);
        slow_ema_factors.resize(num_factors);
        inv_root_factors.resize(num_factors);
        for (int64_t i = 0; i < num_factors; ++i)
        {
            archive.read("fast_ema_" + std::to_string(i), fast_ema_factors[i], true);
            archive.read("slow_ema_" + std::to_string(i), slow_ema_factors[i], true);
            archive.read("inv_root_" + std::to_string(i), inv_root_factors[i], true);
        }
    }
}

std::unique_ptr<torch::optim::OptimizerParamState> FASFAParamState::clone() const
{
    auto cloned = std::make_unique<FASFAParamState>();
    if (step().defined()) cloned->step(step().clone());
    if (exp_avg().defined()) cloned->exp_avg(exp_avg().clone());
    if (exp_avg_sq().defined()) cloned->exp_avg_sq(exp_avg_sq().clone());

    for (const auto& f : fast_ema_factors) cloned->fast_ema_factors.push_back(f.clone());
    for (const auto& f : slow_ema_factors) cloned->slow_ema_factors.push_back(f.clone());
    for (const auto& f : inv_root_factors) cloned->inv_root_factors.push_back(f.clone());
    return cloned;
}


// --- FASFA Implementation ---
FASFA::FASFA(std::vector<torch::Tensor> params, FASFAOptions options)
    : torch::optim::Optimizer(std::move(params), std::make_unique<FASFAOptions>(options))
{
}

torch::Tensor FASFA::step(LossClosure closure)
{
    torch::NoGradGuard no_grad;
    torch::Tensor loss = {};
    if (closure) { loss = closure(); }

    auto& group_options = static_cast<FASFAOptions&>(param_groups_[0].options());

    for (auto& p : param_groups_[0].params())
    {
        if (!p.grad().defined()) { continue; }

        auto grad = p.grad();
        auto& state = static_cast<FASFAParamState&>(*state_.at(p.unsafeGetTensorImpl()));

        if (!state.step().defined())
        {
            state.step(torch::tensor(0.0, torch::dtype(torch::kFloat64).device(torch::kCPU)));
            state.exp_avg(torch::zeros_like(p));
            state.exp_avg_sq(torch::zeros_like(p));
        }
        state.step(state.step() + 1.0);
        double current_step_val = state.step().item<double>();

        if (p.dim() <= 1)
        {
            _fallback_to_adam(p, grad, state, group_options);
            continue;
        }

        // --- Main FASFA Logic ---
        torch::Tensor grad_reshaped;
        if (p.dim() == 2)
        {
            grad_reshaped = grad.t();
        }
        else if (p.dim() == 4)
        {
            grad_reshaped = grad.reshape({p.size(0), -1}).t();
        }
        else
        {
            _fallback_to_adam(p, grad, state, group_options);
            continue;
        }

        // 1. Gather Factor-Wise Statistics (L and R)
        auto L_stat = grad_reshaped.t().matmul(grad_reshaped);
        auto R_stat = grad_reshaped.matmul(grad_reshaped.t());

        if (state.fast_ema_factors.empty())
        {
            state.fast_ema_factors.push_back(torch::zeros_like(L_stat));
            state.fast_ema_factors.push_back(torch::zeros_like(R_stat));
            state.slow_ema_factors.push_back(torch::zeros_like(L_stat));
            state.slow_ema_factors.push_back(torch::zeros_like(R_stat));
            state.inv_root_factors.push_back(torch::eye(L_stat.size(0), L_stat.options()));
            state.inv_root_factors.push_back(torch::eye(R_stat.size(0), R_stat.options()));
        }

        // Update "Factor-Wise" (fast) and "Statistical Factor-Wise" (slow) EMAs
        state.fast_ema_factors[0].mul_(group_options.beta_fast()).add_(L_stat, 1.0 - group_options.beta_fast());
        state.fast_ema_factors[1].mul_(group_options.beta_fast()).add_(R_stat, 1.0 - group_options.beta_fast());
        state.slow_ema_factors[0].mul_(group_options.beta_slow()).add_(L_stat, 1.0 - group_options.beta_slow());
        state.slow_ema_factors[1].mul_(group_options.beta_slow()).add_(R_stat, 1.0 - group_options.beta_slow());

        // 2. Compute Inverse Roots (infrequently)
        if (current_step_val >= group_options.start_preconditioning_step() &&
            static_cast<long>(current_step_val) % group_options.precondition_frequency() == 0)
        {
            // Adaptive damping from the stable "Statistical" factors
            double trace_L = state.slow_ema_factors[0].trace().item<double>();
            double trace_R = state.slow_ema_factors[1].trace().item<double>();
            double dim_L = state.slow_ema_factors[0].size(0);
            double dim_R = state.slow_ema_factors[1].size(0);
            double damping = (dim_L > 0 && dim_R > 0) ? std::sqrt((trace_L * trace_R) / (dim_L * dim_R)) : 1e-6;

            // Compute inverse root of the adaptive "Factor-Wise" (fast) statistics
            state.inv_root_factors[0] = _compute_matrix_inverse_root(state.fast_ema_factors[0], damping,
                                                                     group_options.root_order());
            state.inv_root_factors[1] = _compute_matrix_inverse_root(state.fast_ema_factors[1], damping,
                                                                     group_options.root_order());
        }

        // 3. Precondition Gradient
        auto& L_inv_root = state.inv_root_factors[0];
        auto& R_inv_root = state.inv_root_factors[1];
        auto preconditioned_grad_reshaped = R_inv_root.matmul(grad_reshaped).matmul(L_inv_root);

        torch::Tensor preconditioned_grad;
        if (p.dim() == 2)
        {
            preconditioned_grad = preconditioned_grad_reshaped.t();
        }
        else
        {
            preconditioned_grad = preconditioned_grad_reshaped.t().reshape(p.sizes());
        }

        // 4. Grafting
        torch::Tensor grafted_norm = _compute_grafted_norm(grad, state, group_options);
        torch::Tensor fasfa_norm = preconditioned_grad.norm();

        torch::Tensor final_update = preconditioned_grad;
        if (fasfa_norm.item<double>() > 1e-10)
        {
            final_update.mul_(grafted_norm / fasfa_norm);
        }

        // 5. Weight Decay & Final Parameter Update
        if (group_options.weight_decay() > 0.0)
        {
            p.data().add_(p.data(), -group_options.lr() * group_options.weight_decay());
        }
        p.data().add_(final_update, -group_options.lr());
    }
    return loss;
}

// --- Helper Functions ---
void FASFA::_fallback_to_adam(torch::Tensor& param, const torch::Tensor& grad, FASFAParamState& state,
                              const FASFAOptions& options)
{
    auto& m = state.exp_avg();
    auto& v = state.exp_avg_sq();
    double bias_correction1 = 1.0 - std::pow(options.beta1_adam(), state.step().item<double>());
    double bias_correction2 = 1.0 - std::pow(options.beta2_adam(), state.step().item<double>());

    m.mul_(options.beta1_adam()).add_(grad, 1.0 - options.beta1_adam());
    v.mul_(options.beta2_adam()).addcmul_(grad, grad, 1.0 - options.beta2_adam());

    auto m_hat = m / bias_correction1;
    auto v_hat = v / bias_correction2;
    auto denom = v_hat.sqrt().add_(options.eps());

    if (options.weight_decay() > 0.0)
    {
        param.data().add_(param.data(), -options.lr() * options.weight_decay());
    }
    param.data().addcdiv_(m_hat, denom, -options.lr());
}

torch::Tensor FASFA::_compute_grafted_norm(const torch::Tensor& grad, FASFAParamState& state,
                                           const FASFAOptions& options)
{
    if (options.grafting_type() == "SGD")
    {
        return grad.norm();
    }

    if (options.grafting_type() == "ADAM")
    {
        auto& m = state.exp_avg();
        auto& v = state.exp_avg_sq();
        m.mul_(options.beta1_adam()).add_(grad, 1.0 - options.beta1_adam());
        v.mul_(options.beta2_adam()).addcmul_(grad, grad, 1.0 - options.beta2_adam());
        double bias_correction1 = 1.0 - std::pow(options.beta1_adam(), state.step().item<double>());
        double bias_correction2 = 1.0 - std::pow(options.beta2_adam(), state.step().item<double>());
        auto m_hat = m / bias_correction1;
        auto v_hat = v / bias_correction2;
        return (m_hat / (v_hat.sqrt().add(options.eps()))).norm();
    }

    return torch::tensor(1.0, grad.options());
}

torch::Tensor FASFA::_compute_matrix_inverse_root(const torch::Tensor& matrix, double damping, int root_order)
{
    auto damped_matrix = matrix + torch::eye(matrix.size(0), matrix.options()) * damping;
    std::tuple<torch::Tensor, torch::Tensor> eigh_result;
    try
    {
        //TODO START We should create EIGH
        throw std::runtime_error("We should create EIGH")
        // eigh_result = torch::linalg::eigh(damped_matrix, "U");
        //TODO END We should create EIGH
    }
    catch (const c10::Error& e)
    {
        TORCH_WARN("eigh failed, returning identity. Error: ", e.what());
        return torch::eye(matrix.size(0), matrix.options());
    }
    auto eigenvalues = std::get<0>(eigh_result);
    auto eigenvectors = std::get<1>(eigh_result);
    torch::Tensor inv_root_eigenvalues = eigenvalues.clamp_min(0).pow(-1.0 / static_cast<double>(root_order));
    return eigenvectors.matmul(torch::diag(inv_root_eigenvalues).matmul(eigenvectors.t()));
}

// --- Boilerplate ---
void FASFA::save(torch::serialize::OutputArchive& archive) const { torch::optim::Optimizer::save(archive); }
void FASFA::load(torch::serialize::InputArchive& archive) { torch::optim::Optimizer::load(archive); }

std::unique_ptr<torch::optim::OptimizerParamState> FASFA::make_param_state()
{
    return std::make_unique<FASFAParamState>();
}
