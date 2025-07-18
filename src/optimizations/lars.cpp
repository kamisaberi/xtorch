#include <optimizations/lars.h>
#include <stdexcept>

namespace xt::optim
{
    // --- LARSOptions Methods (Correct, no changes needed) ---
    void LARSOptions::serialize(torch::serialize::OutputArchive& archive) const {
        archive.write("lr", this->lr());
        archive.write("momentum", momentum());
        archive.write("weight_decay", weight_decay());
        archive.write("trust_coefficient", trust_coefficient());
        archive.write("eps", eps());
    }
    void LARSOptions::deserialize(torch::serialize::InputArchive& archive) {
        c10::IValue ivalue;
        if (archive.try_read("lr", ivalue)) { this->lr(ivalue.toDouble()); }
        if (archive.try_read("momentum", ivalue)) { momentum_ = ivalue.toDouble(); }
        if (archive.try_read("weight_decay", ivalue)) { weight_decay_ = ivalue.toDouble(); }
        if (archive.try_read("trust_coefficient", ivalue)) { trust_coefficient_ = ivalue.toDouble(); }
        if (archive.try_read("eps", ivalue)) { eps_ = ivalue.toDouble(); }
    }
    std::unique_ptr<torch::optim::OptimizerOptions> LARSOptions::clone() const {
        auto cloned = std::make_unique<LARSOptions>(this->lr());
        cloned->momentum(momentum()).weight_decay(weight_decay())
              .trust_coefficient(trust_coefficient()).eps(eps());
        return cloned;
    }

    // --- LARSParamState Methods (DEFINITIVELY CORRECTED IMPLEMENTATION) ---
    void LARSParamState::serialize(torch::serialize::OutputArchive& archive) const {
        // Manually serialize each TORCH_ARG member.
        // The key must match the string used in deserialize.
        if (momentum_buffer().defined()) {
            archive.write("momentum_buffer", momentum_buffer(), /*is_buffer=*/true);
        }
    }

    void LARSParamState::deserialize(torch::serialize::InputArchive& archive) {
        // Manually deserialize each TORCH_ARG member.
        at::Tensor temp_momentum;
        if (archive.try_read("momentum_buffer", temp_momentum, /*is_buffer=*/true)) {
            momentum_buffer(temp_momentum);
        }
    }

    std::unique_ptr<torch::optim::OptimizerParamState> LARSParamState::clone() const {
        auto cloned = std::make_unique<LARSParamState>();
        if (momentum_buffer().defined()) {
            cloned->momentum_buffer(momentum_buffer().clone());
        }
        return cloned;
    }


    // --- LARS Implementation (Correct) ---
    LARS::LARS(std::vector<torch::Tensor> params, LARSOptions options)
        : torch::optim::Optimizer(std::move(params), std::make_unique<LARSOptions>(options)) {}

    LARS::LARS(std::vector<torch::Tensor> params, double lr)
        : LARS(std::move(params), LARSOptions(lr)) {}

    torch::Tensor LARS::step(LossClosure closure) {
        torch::NoGradGuard no_grad;
        torch::Tensor loss = {};
        if (closure) { loss = closure(); }

        for (auto& group : param_groups_) {
            auto& options = static_cast<LARSOptions&>(group.options());
            for (auto& p : group.params()) {
                if (!p.grad().defined()) { continue; }

                auto grad = p.grad();
                if (grad.is_sparse()) {
                    throw std::runtime_error("LARS optimizer does not support sparse gradients.");
                }

                auto weight_norm = p.detach().norm(2).item<double>();
                auto grad_norm = grad.norm(2).item<double>();

                double local_lr = 1.0;
                if (weight_norm > 0 && grad_norm > 0) {
                    local_lr = options.trust_coefficient() * weight_norm /
                               (grad_norm + options.weight_decay() * weight_norm + options.eps());
                }

                auto grad_with_wd = grad.add(p.detach(), options.weight_decay());

                auto& state = static_cast<LARSParamState&>(*state_.at(p.unsafeGetTensorImpl()));
                if (!state.momentum_buffer().defined()) {
                    state.momentum_buffer(torch::zeros_like(p));
                }
                auto& momentum_buffer = state.momentum_buffer();
                momentum_buffer.mul_(options.momentum()).add_(grad_with_wd);

                p.data().add_(momentum_buffer, -options.lr() * local_lr);
            }
        }
        return loss;
    }

    // --- Boilerplate Methods ---
    void LARS::save(torch::serialize::OutputArchive& archive) const { torch::optim::Optimizer::save(archive); }
    void LARS::load(torch::serialize::InputArchive& archive) { torch::optim::Optimizer::load(archive); }
    std::unique_ptr<torch::optim::OptimizerParamState> LARS::make_param_state() { return std::make_unique<LARSParamState>(); }
}