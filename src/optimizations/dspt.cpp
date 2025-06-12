#include "include/optimizations/dspt.h"
#include <stdexcept>

// --- DSPTOptions Methods ---
void DSPTOptions::serialize(torch::serialize::OutputArchive& archive) const
{
    archive.write("lr", this->lr);
    archive.write("beta1", beta1());
    archive.write("beta2", beta2());
    archive.write("eps", eps());
    archive.write("weight_decay", weight_decay());
    archive.write("sparsity", sparsity());
    archive.write("prune_rate", prune_rate());
    archive.write("prune_frequency", prune_frequency());
    archive.write("start_pruning_step", start_pruning_step());
    archive.write("low_rank_k", static_cast<int64_t>(low_rank_k()));
}

void DSPTOptions::deserialize(torch::serialize::InputArchive& archive)
{
    c10::IValue ivalue;
    if (archive.try_read("lr", ivalue)) { this->lr = ivalue.toDouble(); }
    if (archive.try_read("beta1", ivalue)) { beta1_ = ivalue.toDouble(); }
    if (archive.try_read("beta2", ivalue)) { beta2_ = ivalue.toDouble(); }
    if (archive.try_read("eps", ivalue)) { eps_ = ivalue.toDouble(); }
    if (archive.try_read("weight_decay", ivalue)) { weight_decay_ = ivalue.toDouble(); }
    if (archive.try_read("sparsity", ivalue)) { sparsity_ = ivalue.toDouble(); }
    if (archive.try_read("prune_rate", ivalue)) { prune_rate_ = ivalue.toDouble(); }
    if (archive.try_read("prune_frequency", ivalue)) { prune_frequency_ = ivalue.toInt(); }
    if (archive.try_read("start_pruning_step", ivalue)) { start_pruning_step_ = ivalue.toInt(); }
    if (archive.try_read("low_rank_k", ivalue)) { low_rank_k_ = ivalue.toInt(); }
}

std::unique_ptr<torch::optim::OptimizerOptions> DSPTOptions::clone() const
{
    auto cloned = std::make_unique<DSPTOptions>(this->lr);
    cloned->beta1(beta1()).beta2(beta2()).eps(eps()).weight_decay(weight_decay())
          .sparsity(sparsity()).prune_rate(prune_rate())
          .prune_frequency(prune_frequency()).start_pruning_step(start_pruning_step())
          .low_rank_k(low_rank_k());
    return cloned;
}

// --- DSPTParamState Methods ---
void DSPTParamState::serialize(torch::serialize::OutputArchive& archive) const
{
    archive.write("step", step(), true);
    archive.write("exp_avg", exp_avg(), true);
    archive.write("exp_avg_sq", exp_avg_sq(), true);
    archive.write("mask", mask(), true);
}

void DSPTParamState::deserialize(torch::serialize::InputArchive& archive)
{
    at::Tensor temp;
    if (archive.try_read("step", temp, true)) { step_ = temp; }
    if (archive.try_read("exp_avg", temp, true)) { exp_avg_ = temp; }
    if (archive.try_read("exp_avg_sq", temp, true)) { exp_avg_sq_ = temp; }
    if (archive.try_read("mask", temp, true)) { mask_ = temp; }
}

std::unique_ptr<torch::optim::OptimizerParamState> DSPTParamState::clone() const
{
    auto cloned = std::make_unique<DSPTParamState>();
    if (step().defined()) cloned->step(step().clone());
    if (exp_avg().defined()) cloned->exp_avg(exp_avg().clone());
    if (exp_avg_sq().defined()) cloned->exp_avg_sq(exp_avg_sq().clone());
    if (mask().defined()) cloned->mask(mask().clone());
    return cloned;
}

// --- DSPT Implementation ---
DSPT::DSPT(std::vector<torch::Tensor> params, DSPTOptions options)
    : torch::optim::Optimizer(std::move(params), std::make_unique<DSPTOptions>(options))
{
}

torch::Tensor DSPT::step(LossClosure closure)
{
    torch::NoGradGuard no_grad;
    torch::Tensor loss = {};
    if (closure) { loss = closure(); }

    for (auto& group : param_groups_)
    {
        auto& options = static_cast<DSPTOptions&>(group.options());
        for (auto& p : group.params())
        {
            if (!p.grad().defined()) { continue; }

            auto full_grad = p.grad();
            auto& state = static_cast<DSPTParamState&>(*state_.at(p.unsafeGetTensorImpl()));

            // Initialize state
            if (!state.step().defined())
            {
                state.step(torch::tensor(0.0, torch::dtype(torch::kFloat64).device(torch::kCPU)));
                state.exp_avg(torch::zeros_like(p));
                state.exp_avg_sq(torch::zeros_like(p));
                state.mask(torch::ones_like(p)); // Start dense
            }
            state.step(state.step() + 1.0);
            double current_step_val = state.step().item<double>();

            // 1. Dynamic Pruning and Growing (if scheduled)
            if (current_step_val >= options.start_pruning_step() &&
                static_cast<long>(current_step_val) % options.prune_frequency() == 0)
            {
                _update_mask(p, full_grad, state, options);
            }

            // 2. Apply mask to gradient for the sparse update
            auto masked_grad = full_grad * state.mask();

            // 3. Base Adam-like update on the sparse weights
            auto& exp_avg = state.exp_avg();
            auto& exp_avg_sq = state.exp_avg_sq();
            exp_avg.mul_(options.beta1()).add_(masked_grad, 1.0 - options.beta1());
            exp_avg_sq.mul_(options.beta2()).addcmul_(masked_grad, masked_grad, 1.0 - options.beta2());

            double bias_correction1 = 1.0 - std::pow(options.beta1(), current_step_val);
            double bias_correction2 = 1.0 - std::pow(options.beta2(), current_step_val);

            auto m_hat = exp_avg / bias_correction1;
            auto v_hat = exp_avg_sq / bias_correction2;
            auto denom = v_hat.sqrt().add_(options.eps());

            p.data().addcdiv_(m_hat, denom, -options.lr);

            // 4. Low-Rank update (if enabled)
            if (options.low_rank_k() > 0)
            {
                _apply_low_rank_update(p, state, options);
            }

            // 5. Ensure pruned weights remain zero after all updates
            p.data().mul_(state.mask());
        }
    }
    return loss;
}

void DSPT::_update_mask(
    torch::Tensor& param,
    const torch::Tensor& full_grad,
    DSPTParamState& state,
    const DSPTOptions& options)
{
    // --- Pruning: Remove weights with smallest magnitude ---
    auto current_mask = state.mask();
    auto num_dense = current_mask.sum().item<int64_t>();
    auto num_to_prune = static_cast<int64_t>(options.prune_rate() * num_dense);

    if (num_to_prune > 0)
    {
        // Find scores of active weights (magnitude)
        auto scores = param.detach().abs();
        scores.masked_fill_(current_mask == 0, std::numeric_limits<float>::infinity()); // Ignore already pruned weights

        // Find the pruning threshold
        auto threshold = std::get<0>(torch::kthvalue(scores.flatten(), num_to_prune));

        // Create a new mask by pruning weights below the threshold
        auto new_mask = current_mask * (scores > threshold).to(current_mask.dtype());
        state.mask(new_mask);
    }

    // --- Growing: Add new weights with largest gradient magnitude ---
    auto num_to_grow = num_to_prune; // Grow the same number we pruned
    if (num_to_grow > 0)
    {
        // Find scores of inactive weights (gradient magnitude)
        auto grad_scores = full_grad.detach().abs();
        grad_scores.masked_fill_(state.mask() == 1, 0.0); // Ignore already active weights

        // Find the growth threshold
        auto growth_threshold = std::get<0>(torch::kthvalue(grad_scores.flatten(), grad_scores.numel() - num_to_grow));

        // Add new connections to the mask
        auto growth_mask = (grad_scores >= growth_threshold).to(state.mask().dtype());
        state.mask((state.mask() + growth_mask).clamp_max_(1.0));
    }
}

void DSPT::_apply_low_rank_update(
    torch::Tensor& param,
    DSPTParamState& state,
    const DSPTOptions& options)
{
    if (param.dim() != 2)
    {
        // For simplicity, this implementation only supports low-rank updates for 2D tensors (e.g., Linear layers).
        // A more advanced version would matricize Conv kernels.
        return;
    }

    // Use the momentum tensor as an approximation of the weight change
    auto momentum = state.exp_avg();


    //TODO START We should create SVD
    throw std::runtime_error("torch::linalg::svd does not exist in libtorch");
    // SVD: M = U S V^T

    // auto svd_result = torch::linalg::svd(momentum, false); // false = do not compute full U/V matrices
    // auto U = std::get<0>(svd_result);
    // auto S = std::get<1>(svd_result);
    // auto Vh = std::get<2>(svd_result);

    // int k = options.low_rank_k();

    // // Reconstruct the low-rank approximation
    // auto U_k = U.slice(/*dim=*/1, 0, k);
    // auto S_k = torch::diag(S.slice(/*dim=*/0, 0, k));
    // auto Vh_k = Vh.slice(/*dim=*/0, 0, k);
    //
    // auto low_rank_update = U_k.matmul(S_k).matmul(Vh_k);
    //
    // // Apply the dense low-rank update to the parameter
    // param.data().add_(low_rank_update, -options.lr());
    //TODO END We should create SVD
}

// --- Boilerplate Methods ---
void DSPT::save(torch::serialize::OutputArchive& archive) const
{
    torch::optim::Optimizer::save(archive);
}

void DSPT::load(torch::serialize::InputArchive& archive)
{
    torch::optim::Optimizer::load(archive);
}

std::unique_ptr<torch::optim::OptimizerParamState> DSPT::make_param_state()
{
    return std::make_unique<DSPTParamState>();
}
