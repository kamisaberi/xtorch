#include "include/optimizations/qhm.h"
#include <stdexcept>

namespace xt::optim
{
    // --- QHMOptions Methods ---
    void QHMOptions::serialize(torch::serialize::OutputArchive& archive) const {
        archive.write("lr", this->lr());
        archive.write("beta", beta());
        archive.write("nu", nu());
        archive.write("weight_decay", weight_decay());
    }
    void QHMOptions::deserialize(torch::serialize::InputArchive& archive) {
        c10::IValue ivalue;
        if (archive.try_read("lr", ivalue)) { this->lr(ivalue.toDouble()); }
        if (archive.try_read("beta", ivalue)) { beta_ = ivalue.toDouble(); }
        if (archive.try_read("nu", ivalue)) { nu_ = ivalue.toDouble(); }
        if (archive.try_read("weight_decay", ivalue)) { weight_decay_ = ivalue.toDouble(); }
    }
    std::unique_ptr<torch::optim::OptimizerOptions> QHMOptions::clone() const {
        auto cloned = std::make_unique<QHMOptions>(this->lr());
        cloned->beta(beta()).nu(nu()).weight_decay(weight_decay());
        return cloned;
    }

    // --- QHMParamState Methods ---
    void QHMParamState::serialize(torch::serialize::OutputArchive& archive) const {
        if(momentum_buffer().defined()) archive.write("momentum_buffer", momentum_buffer(), true);
    }
    void QHMParamState::deserialize(torch::serialize::InputArchive& archive) {
        at::Tensor temp;
        if(archive.try_read("momentum_buffer", temp, true)) { momentum_buffer_ = temp; }
    }
    std::unique_ptr<torch::optim::OptimizerParamState> QHMParamState::clone() const {
        auto cloned = std::make_unique<QHMParamState>();
        if(momentum_buffer().defined()) cloned->momentum_buffer(momentum_buffer().clone());
        return cloned;
    }

    // --- QHM Implementation ---
    QHM::QHM(std::vector<torch::Tensor> params, QHMOptions options)
        : torch::optim::Optimizer(std::move(params), std::make_unique<QHMOptions>(options)) {}

    QHM::QHM(std::vector<torch::Tensor> params, double lr)
        : QHM(std::move(params), QHMOptions(lr)) {}

    torch::Tensor QHM::step(LossClosure closure) {
        torch::NoGradGuard no_grad;
        torch::Tensor loss = {};
        if (closure) { loss = closure(); }

        auto& group_options = static_cast<QHMOptions&>(param_groups_[0].options());

        for (auto& group : param_groups_) {
            for (auto& p : group.params()) {
                if (!p.grad().defined()) { continue; }

                auto grad = p.grad();
                if (grad.is_sparse()) {
                    throw std::runtime_error("QHM optimizer does not support sparse gradients.");
                }

                // Apply classic weight decay (L2 regularization)
                if (group_options.weight_decay() > 0.0) {
                    grad = grad.add(p.detach(), group_options.weight_decay());
                }

                auto& state = static_cast<QHMParamState&>(*state_.at(p.unsafeGetTensorImpl()));
                if (!state.momentum_buffer().defined()) {
                    // Initialize momentum with the first gradient
                    state.momentum_buffer(grad.clone());
                }

                auto& m = state.momentum_buffer();
                double beta = group_options.beta();
                double nu = group_options.nu();

                // 1. Update the momentum EMA
                // m_t = beta * m_{t-1} + (1 - beta) * g_t
                m.mul_(beta).add_(grad, 1.0 - beta);

                // 2. Construct the quasi-hyperbolic update
                // update = (1 - nu) * g_t + nu * m_t
                auto update = grad * (1.0 - nu) + m * nu;

                // 3. Final update
                p.data().add_(update, -group_options.lr());
            }
        }
        return loss;
    }

    // --- Boilerplate Methods ---
    void QHM::save(torch::serialize::OutputArchive& archive) const { torch::optim::Optimizer::save(archive); }
    void QHM::load(torch::serialize::InputArchive& archive) { torch::optim::Optimizer::load(archive); }
    std::unique_ptr<torch::optim::OptimizerParamState> QHM::make_param_state() { return std::make_unique<QHMParamState>(); }
}