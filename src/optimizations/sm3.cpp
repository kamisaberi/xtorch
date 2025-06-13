#include "include/optimizations/sm3.h"
#include <stdexcept>

// --- SM3Options Methods ---
void SM3Options::serialize(torch::serialize::OutputArchive& archive) const {
    archive.write("lr", this->lr());
    archive.write("beta", beta());
    archive.write("eps", eps());
}
void SM3Options::deserialize(torch::serialize::InputArchive& archive) {
    c10::IValue ivalue;
    if (archive.try_read("lr", ivalue)) { this->lr(ivalue.toDouble()); }
    if (archive.try_read("beta", ivalue)) { beta_ = ivalue.toDouble(); }
    if (archive.try_read("eps", ivalue)) { eps_ = ivalue.toDouble(); }
}
std::unique_ptr<torch::optim::OptimizerOptions> SM3Options::clone() const {
    auto cloned = std::make_unique<SM3Options>(this->lr());
    cloned->beta(beta()).eps(eps());
    return cloned;
}

// --- SM3ParamState Methods ---
void SM3ParamState::serialize(torch::serialize::OutputArchive& archive) const {
    if(momentum_buffer().defined()) archive.write("momentum_buffer", momentum_buffer(), true);
    if(row_accumulator.defined()) archive.write("row_accumulator", row_accumulator, true);
    if(col_accumulator.defined()) archive.write("col_accumulator", col_accumulator, true);
}
void SM3ParamState::deserialize(torch::serialize::InputArchive& archive) {
    at::Tensor temp;
    if(archive.try_read("momentum_buffer", temp, true)) { momentum_buffer_ = temp; }
    if(archive.try_read("row_accumulator", temp, true)) { row_accumulator = temp; }
    if(archive.try_read("col_accumulator", temp, true)) { col_accumulator = temp; }
}
std::unique_ptr<torch::optim::OptimizerParamState> SM3ParamState::clone() const {
    auto cloned = std::make_unique<SM3ParamState>();
    if(momentum_buffer().defined()) cloned->momentum_buffer(momentum_buffer().clone());
    if(row_accumulator.defined()) cloned->row_accumulator = row_accumulator.clone();
    if(col_accumulator.defined()) cloned->col_accumulator = col_accumulator.clone();
    return cloned;
}


// --- SM3 Implementation ---
SM3::SM3(std::vector<torch::Tensor> params, SM3Options options)
    : torch::optim::Optimizer(std::move(params), std::make_unique<SM3Options>(options)) {}

SM3::SM3(std::vector<torch::Tensor> params, double lr)
    : SM3(std::move(params), SM3Options(lr)) {}

torch::Tensor SM3::step(LossClosure closure) {
    torch::NoGradGuard no_grad;
    torch::Tensor loss = {};
    if (closure) { loss = closure(); }

    auto& group_options = static_cast<SM3Options&>(param_groups_[0].options());

    for (auto& group : param_groups_) {
        for (auto& p : group.params()) {
            if (!p.grad().defined()) { continue; }

            auto grad = p.grad();
            if (grad.is_sparse()) {
                throw std::runtime_error("SM3 optimizer does not support sparse gradients.");
            }

            auto& state = static_cast<SM3ParamState&>(*state_.at(p.unsafeGetTensorImpl()));
            if (!state.momentum_buffer().defined()) {
                state.momentum_buffer(torch::zeros_like(p));
            }

            // Apply momentum
            auto& momentum_buffer = state.momentum_buffer();
            momentum_buffer.mul_(group_options.beta()).add_(grad, 1.0 - group_options.beta());

            // --- SM3 Adaptive LR Logic ---
            torch::Tensor update;
            if (p.dim() < 2) {
                // Fallback for 1D tensors (biases, etc.): Use AdaGrad-like update
                if (!state.row_accumulator.defined()) {
                    state.row_accumulator = torch::zeros_like(p);
                }
                auto& accum = state.row_accumulator;
                accum.addcmul_(grad, grad, 1.0); // accum += grad^2
                auto denom = accum.sqrt().add_(group_options.eps());
                update = momentum_buffer / denom;

            } else {
                // Main SM3 logic for 2D+ tensors
                auto grad_reshaped = (p.dim() > 2) ? grad.reshape({p.size(0), -1}) : grad;
                int M = grad_reshaped.size(0);
                int N = grad_reshaped.size(1);

                if (!state.row_accumulator.defined()) {
                    state.row_accumulator = torch::zeros({M}, p.options());
                    state.col_accumulator = torch::zeros({N}, p.options());
                }
                auto& v_row = state.row_accumulator;
                auto& v_col = state.col_accumulator;

                // 1. Update row and column accumulators
                // v_row = min(v_row_prev, max(G^2, axis=1))
                auto grad_sq = grad_reshaped.square();
                v_row = torch::min(v_row, std::get<0>(torch::max(grad_sq, 1)));
                v_col = torch::min(v_col, std::get<0>(torch::max(grad_sq, 0)));

                // 2. Construct the adaptive learning rate denominator
                // Denom = sqrt(outer_product(v_row, v_col))
                // For memory efficiency, we do this division row-by-row and col-by-col
                auto momentum_reshaped = (p.dim() > 2) ? momentum_buffer.reshape({M, N}) : momentum_buffer;
                auto update_reshaped = momentum_reshaped / (v_row.unsqueeze(1).sqrt() + group_options.eps());
                update_reshaped = update_reshaped / (v_col.unsqueeze(0).sqrt() + group_options.eps());

                update = (p.dim() > 2) ? update_reshaped.reshape(p.sizes()) : update_reshaped;
            }

            // Apply the final update
            p.data().add_(update, -group_options.lr());
        }
    }
    return loss;
}

// --- Boilerplate Methods ---
void SM3::save(torch::serialize::OutputArchive& archive) const { torch::optim::Optimizer::save(archive); }
void SM3::load(torch::serialize::InputArchive& archive) { torch::optim::Optimizer::load(archive); }
std::unique_ptr<torch::optim::OptimizerParamState> SM3::make_param_state() { return std::make_unique<SM3ParamState>(); }