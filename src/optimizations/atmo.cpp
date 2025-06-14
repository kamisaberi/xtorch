#include "include/optimizations/atmo.h"
#include <stdexcept>

// --- ATMOOptions Methods ---
void ATMOOptions::serialize(torch::serialize::OutputArchive& archive) const {
    archive.write("lr", this->lr());
    archive.write("weight_decay", weight_decay());
    archive.write("temperature", temperature());
    archive.write("eps", eps());
    archive.write("betas_size", static_cast<int64_t>(betas_.size()));
    for (size_t i = 0; i < betas_.size(); ++i) {
        archive.write("beta_" + std::to_string(i), betas_[i]);
    }
}
void ATMOOptions::deserialize(torch::serialize::InputArchive& archive) {
    c10::IValue ivalue;
    if (archive.try_read("lr", ivalue)) { this->lr(ivalue.toDouble()); }
    if (archive.try_read("weight_decay", ivalue)) { weight_decay_ = ivalue.toDouble(); }
    if (archive.try_read("temperature", ivalue)) { temperature_ = ivalue.toDouble(); }
    if (archive.try_read("eps", ivalue)) { eps_ = ivalue.toDouble(); }
    if (archive.try_read("betas_size", ivalue)) {
        betas_.resize(ivalue.toInt());
        for (size_t i = 0; i < betas_.size(); ++i) {
            if (archive.try_read("beta_" + std::to_string(i), ivalue)) {
                betas_[i] = ivalue.toDouble();
            }
        }
    }
}
std::unique_ptr<torch::optim::OptimizerOptions> ATMOOptions::clone() const {
    auto cloned = std::make_unique<ATMOOptions>(this->lr());
    cloned->betas(this->betas());
    cloned->weight_decay(weight_decay()).temperature(temperature()).eps(eps());
    return cloned;
}

// --- ATMOParamState Methods --- (Identical to AggMo's)
void ATMOParamState::serialize(torch::serialize::OutputArchive& archive) const {
    archive.write("num_momentum_buffers", static_cast<int64_t>(momentum_buffers.size()));
    for (size_t i = 0; i < momentum_buffers.size(); ++i) {
        if(momentum_buffers[i].defined())
            archive.write("momentum_buffer_" + std::to_string(i), momentum_buffers[i], true);
    }
}
void ATMOParamState::deserialize(torch::serialize::InputArchive& archive) {
    c10::IValue ivalue;
    if (archive.try_read("num_momentum_buffers", ivalue)) {
        momentum_buffers.resize(ivalue.toInt());
        for (size_t i = 0; i < momentum_buffers.size(); ++i) {
            at::Tensor temp;
            if (archive.try_read("momentum_buffer_" + std::to_string(i), temp, true)) {
                 momentum_buffers[i] = temp;
            }
        }
    }
}
std::unique_ptr<torch::optim::OptimizerParamState> ATMOParamState::clone() const {
    auto cloned = std::make_unique<ATMOParamState>();
    for(const auto& buf : momentum_buffers) {
        if (buf.defined()) cloned->momentum_buffers.push_back(buf.clone());
        else cloned->momentum_buffers.push_back(torch::Tensor());
    }
    return cloned;
}

// --- ATMO Implementation ---
ATMO::ATMO(std::vector<torch::Tensor> params, ATMOOptions options)
    : torch::optim::Optimizer(std::move(params), std::make_unique<ATMOOptions>(options)) {
    TORCH_CHECK(!options.betas().empty(), "ATMO requires at least one beta value.");
}

ATMO::ATMO(std::vector<torch::Tensor> params, double lr)
    : ATMO(std::move(params), ATMOOptions(lr)) {}

torch::Tensor ATMO::step(LossClosure closure) {
    torch::NoGradGuard no_grad;
    torch::Tensor loss = {};
    if (closure) { loss = closure(); }

    auto& group_options = static_cast<ATMOOptions&>(param_groups_[0].options());
    const auto& betas = group_options.betas();
    int num_betas = betas.size();

    for (auto& group : param_groups_) {
        for (auto& p : group.params()) {
            if (!p.grad().defined()) { continue; }

            auto grad = p.grad();
            if (grad.is_sparse()) {
                throw std::runtime_error("ATMO optimizer does not support sparse gradients.");
            }

            if (group_options.weight_decay() > 0.0) {
                grad = grad.add(p.detach(), group_options.weight_decay());
            }

            auto& state = static_cast<ATMOParamState&>(*state_.at(p.unsafeGetTensorImpl()));

            if (state.momentum_buffers.empty()) {
                state.momentum_buffers.resize(num_betas);
                for (int i = 0; i < num_betas; ++i) {
                    state.momentum_buffers[i] = torch::zeros_like(p);
                }
            }

            // 1. Calculate alignment scores for each momentum buffer
            std::vector<double> alignment_scores;
            auto grad_norm = grad.norm().item<double>();

            for (int i = 0; i < num_betas; ++i) {
                auto& buf = state.momentum_buffers[i];
                auto buf_norm = buf.norm().item<double>();

                if (grad_norm < group_options.eps() || buf_norm < group_options.eps()) {
                    alignment_scores.push_back(0.0);
                } else {
                    auto cosine_sim = (grad * buf).sum().item<double>() / (grad_norm * buf_norm);
                    alignment_scores.push_back(cosine_sim);
                }
            }

            // 2. Compute softmax attention weights from scores
            auto scores_tensor = torch::from_blob(alignment_scores.data(), {(long)alignment_scores.size()}, torch::kFloat64).clone();
            auto attention_weights = torch::softmax(scores_tensor / group_options.temperature(), 0);

            // 3. Update all momentum buffers and compute the weighted average
            torch::Tensor aggregated_momentum = torch::zeros_like(p);
            for (int i = 0; i < num_betas; ++i) {
                auto& buf = state.momentum_buffers[i];
                double beta_i = betas[i];

                // Update the momentum buffer: buf = beta*buf + grad
                buf.mul_(beta_i).add_(grad);

                // Add to the sum, weighted by attention
                aggregated_momentum.add_(buf, attention_weights[i].item<double>());
            }

            // 4. Final parameter update
            p.data().add_(aggregated_momentum, -group_options.lr());
        }
    }
    return loss;
}

// --- Boilerplate Methods ---
void ATMO::save(torch::serialize::OutputArchive& archive) const { torch::optim::Optimizer::save(archive); }
void ATMO::load(torch::serialize::InputArchive& archive) { torch::optim::Optimizer::load(archive); }
std::unique_ptr<torch::optim::OptimizerParamState> ATMO::make_param_state() { return std::make_unique<ATMOParamState>(); }