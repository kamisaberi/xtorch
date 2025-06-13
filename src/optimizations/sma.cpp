#include "include/optimizations/sma.h"
#include <stdexcept>

// --- SMAOptions Methods ---
void SMAOptions::serialize(torch::serialize::OutputArchive& archive) const {
    archive.write("lr", this->lr());
    archive.write("window_size", window_size());
    archive.write("eps", eps());
    archive.write("weight_decay", weight_decay());
}
void SMAOptions::deserialize(torch::serialize::InputArchive& archive) {
    c10::IValue ivalue;
    if (archive.try_read("lr", ivalue)) { this->lr(ivalue.toDouble()); }
    if (archive.try_read("window_size", ivalue)) { window_size_ = ivalue.toInt(); }
    if (archive.try_read("eps", ivalue)) { eps_ = ivalue.toDouble(); }
    if (archive.try_read("weight_decay", ivalue)) { weight_decay_ = ivalue.toDouble(); }
}
std::unique_ptr<torch::optim::OptimizerOptions> SMAOptions::clone() const {
    auto cloned = std::make_unique<SMAOptions>(this->lr());
    cloned->window_size(window_size()).eps(eps()).weight_decay(weight_decay());
    return cloned;
}

// --- SMAParamState Methods ---
void SMAParamState::serialize(torch::serialize::OutputArchive& archive) const {
    // TORCH_ARG members are handled by base class, but we have custom members.
    // The base Optimizer's save/load doesn't handle custom members automatically.
    // We must serialize them ourselves.
    archive.write("grad_sum", grad_sum(), true);
    archive.write("grad_sq_sum", grad_sq_sum(), true);

    archive.write("window_current_size", static_cast<int64_t>(grad_window.size()));
    for (size_t i = 0; i < grad_window.size(); ++i) {
        archive.write("grad_win_" + std::to_string(i), grad_window[i], true);
        archive.write("grad_sq_win_" + std::to_string(i), grad_sq_window[i], true);
    }
}
void SMAParamState::deserialize(torch::serialize::InputArchive& archive) {
    at::Tensor temp;
    if(archive.try_read("grad_sum", temp, true)) { grad_sum_ = temp; }
    if(archive.try_read("grad_sq_sum", temp, true)) { grad_sq_sum_ = temp; }

    c10::IValue ivalue;
    if (archive.try_read("window_current_size", ivalue)) {
        int64_t current_size = ivalue.toInt();
        grad_window.resize(current_size);
        grad_sq_window.resize(current_size);
        for (int64_t i = 0; i < current_size; ++i) {
            archive.read("grad_win_" + std::to_string(i), grad_window[i], true);
            archive.read("grad_sq_win_" + std::to_string(i), grad_sq_window[i], true);
        }
    }
}
std::unique_ptr<torch::optim::OptimizerParamState> SMAParamState::clone() const {
    auto cloned = std::make_unique<SMAParamState>();
    if(grad_sum().defined()) cloned->grad_sum(grad_sum().clone());
    if(grad_sq_sum().defined()) cloned->grad_sq_sum(grad_sq_sum().clone());
    for(const auto& g : grad_window) cloned->grad_window.push_back(g.clone());
    for(const auto& g_sq : grad_sq_window) cloned->grad_sq_window.push_back(g_sq.clone());
    return cloned;
}


// --- SMA Implementation ---
SMA::SMA(std::vector<torch::Tensor> params, SMAOptions options)
    : torch::optim::Optimizer(std::move(params), std::make_unique<SMAOptions>(options)) {}

SMA::SMA(std::vector<torch::Tensor> params, double lr)
    : SMA(std::move(params), SMAOptions(lr)) {}

torch::Tensor SMA::step(LossClosure closure) {
    torch::NoGradGuard no_grad;
    torch::Tensor loss = {};
    if (closure) { loss = closure(); }

    auto& group_options = static_cast<SMAOptions&>(param_groups_[0].options());

    for (auto& group : param_groups_) {
        for (auto& p : group.params()) {
            if (!p.grad().defined()) { continue; }

            auto grad = p.grad();
            if (grad.is_sparse()) {
                throw std::runtime_error("SMA optimizer does not support sparse gradients.");
            }

            auto& state = static_cast<SMAParamState&>(*state_.at(p.unsafeGetTensorImpl()));

            // Initialize state
            if (!state.grad_sum().defined()) {
                state.grad_sum(torch::zeros_like(p));
                state.grad_sq_sum(torch::zeros_like(p));
            }

            // Apply decoupled weight decay
            if (group_options.weight_decay() > 0.0) {
                p.data().mul_(1.0 - group_options.lr() * group_options.weight_decay());
            }

            auto& grad_sum = state.grad_sum();
            auto& grad_sq_sum = state.grad_sq_sum();
            auto& grad_window = state.grad_window;
            auto& grad_sq_window = state.grad_sq_window;

            // 1. Update the sliding window and sums
            auto grad_sq = grad.square();

            // Add new gradient and its square to the window and sum
            grad_window.push_back(grad.clone());
            grad_sq_window.push_back(grad_sq.clone());
            grad_sum.add_(grad);
            grad_sq_sum.add_(grad_sq);

            // If window is full, remove the oldest gradient and its square
            if (grad_window.size() > group_options.window_size()) {
                grad_sum.sub_(grad_window.front());
                grad_sq_sum.sub_(grad_sq_window.front());
                grad_window.pop_front();
                grad_sq_window.pop_front();
            }

            // 2. Compute the Simple Moving Averages
            double current_window_size = grad_window.size();
            auto m_sma = grad_sum / current_window_size;
            auto v_sma = grad_sq_sum / current_window_size;

            // 3. Compute the final update (Adam-like denominator)
            auto denom = v_sma.sqrt().add_(group_options.eps());

            p.data().addcdiv_(m_sma, denom, -group_options.lr());
        }
    }
    return loss;
}

// --- Boilerplate Methods ---
void SMA::save(torch::serialize::OutputArchive& archive) const { torch::optim::Optimizer::save(archive); }
void SMA::load(torch::serialize::InputArchive& archive) { torch::optim::Optimizer::load(archive); }
std::unique_ptr<torch::optim::OptimizerParamState> SMA::make_param_state() { return std::make_unique<SMAParamState>(); }