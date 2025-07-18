#include <optimizations/agg_mo.h>
#include <stdexcept>
namespace xt::optim
{
    // --- AggMoOptions Methods ---
    void AggMoOptions::serialize(torch::serialize::OutputArchive& archive) const {
        archive.write("lr", this->lr());
        archive.write("weight_decay", weight_decay());
        // Serialize std::vector<double>
        archive.write("betas_size", static_cast<int64_t>(betas_.size()));
        for (size_t i = 0; i < betas_.size(); ++i) {
            archive.write("beta_" + std::to_string(i), betas_[i]);
        }
    }
    void AggMoOptions::deserialize(torch::serialize::InputArchive& archive) {
        c10::IValue ivalue;
        if (archive.try_read("lr", ivalue)) { this->lr(ivalue.toDouble()); }
        if (archive.try_read("weight_decay", ivalue)) { weight_decay_ = ivalue.toDouble(); }

        if (archive.try_read("betas_size", ivalue)) {
            int64_t betas_size = ivalue.toInt();
            betas_.resize(betas_size);
            for (int64_t i = 0; i < betas_size; ++i) {
                if (archive.try_read("beta_" + std::to_string(i), ivalue)) {
                    betas_[i] = ivalue.toDouble();
                } else {
                    TORCH_WARN("Could not read beta_", i, " for AggMoOptions");
                }
            }
        }
    }
    std::unique_ptr<torch::optim::OptimizerOptions> AggMoOptions::clone() const {
        auto cloned = std::make_unique<AggMoOptions>(this->lr());
        cloned->betas(this->betas()); // Copy the vector
        cloned->weight_decay(weight_decay());
        return cloned;
    }

    // --- AggMoParamState Methods ---
    void AggMoParamState::serialize(torch::serialize::OutputArchive& archive) const {
        archive.write("num_momentum_buffers", static_cast<int64_t>(momentum_buffers.size()));
        for (size_t i = 0; i < momentum_buffers.size(); ++i) {
            if(momentum_buffers[i].defined())
                archive.write("momentum_buffer_" + std::to_string(i), momentum_buffers[i], true);
        }
    }
    void AggMoParamState::deserialize(torch::serialize::InputArchive& archive) {
        c10::IValue ivalue;
        if (archive.try_read("num_momentum_buffers", ivalue)) {
            int64_t num_buffers = ivalue.toInt();
            momentum_buffers.resize(num_buffers);
            for (int64_t i = 0; i < num_buffers; ++i) {
                at::Tensor temp;
                // It's possible a buffer was not saved if it wasn't defined, handle this
                if (archive.try_read("momentum_buffer_" + std::to_string(i), temp, true)) {
                    momentum_buffers[i] = temp;
                }
            }
        }
    }
    std::unique_ptr<torch::optim::OptimizerParamState> AggMoParamState::clone() const {
        auto cloned = std::make_unique<AggMoParamState>();
        for(const auto& buf : momentum_buffers) {
            if (buf.defined()) cloned->momentum_buffers.push_back(buf.clone());
            else cloned->momentum_buffers.push_back(torch::Tensor()); // Keep undefined if original was
        }
        return cloned;
    }


    // --- AggMo Implementation ---
    AggMo::AggMo(std::vector<torch::Tensor> params, AggMoOptions options)
        : torch::optim::Optimizer(std::move(params), std::make_unique<AggMoOptions>(options)) {
        TORCH_CHECK(!options.betas().empty(), "AggMo requires at least one beta value.");
    }

    AggMo::AggMo(std::vector<torch::Tensor> params, double lr)
        : AggMo(std::move(params), AggMoOptions(lr)) {}

    torch::Tensor AggMo::step(LossClosure closure) {
        torch::NoGradGuard no_grad;
        torch::Tensor loss = {};
        if (closure) { loss = closure(); }

        auto& group_options = static_cast<AggMoOptions&>(param_groups_[0].options());
        const auto& betas = group_options.betas();
        int num_betas = betas.size();

        for (auto& group : param_groups_) {
            for (auto& p : group.params()) {
                if (!p.grad().defined()) { continue; }

                auto grad = p.grad();
                if (grad.is_sparse()) {
                    throw std::runtime_error("AggMo optimizer does not support sparse gradients.");
                }

                // Apply classic L2 regularization (weight decay)
                if (group_options.weight_decay() > 0.0) {
                    grad = grad.add(p.detach(), group_options.weight_decay());
                }

                auto& state = static_cast<AggMoParamState&>(*state_.at(p.unsafeGetTensorImpl()));

                // Initialize momentum buffers if this is the first step for this parameter
                if (state.momentum_buffers.empty()) {
                    state.momentum_buffers.resize(num_betas);
                    for (int i = 0; i < num_betas; ++i) {
                        state.momentum_buffers[i] = torch::zeros_like(p);
                    }
                }

                torch::Tensor aggregated_momentum = torch::zeros_like(p);

                // 1. Update each momentum buffer and sum them up
                for (int i = 0; i < num_betas; ++i) {
                    auto& buf = state.momentum_buffers[i];
                    double beta_i = betas[i];

                    // Standard momentum update: buf = beta * buf + grad
                    // (Note: AggMo paper's formula is often written as beta*buf + (1-beta)*grad for EMA,
                    //  but using beta*buf + grad is also a valid momentum variant.
                    //  Let's use the common SGD momentum form: buf = beta*buf + grad)
                    buf.mul_(beta_i).add_(grad);

                    aggregated_momentum.add_(buf);
                }

                // 2. Average the aggregated momentums
                if (num_betas > 0) {
                    aggregated_momentum.div_(num_betas);
                }

                // 3. Final parameter update
                p.data().add_(aggregated_momentum, -group_options.lr());
            }
        }
        return loss;
    }

    // --- Boilerplate Methods ---
    void AggMo::save(torch::serialize::OutputArchive& archive) const { torch::optim::Optimizer::save(archive); }
    void AggMo::load(torch::serialize::InputArchive& archive) { torch::optim::Optimizer::load(archive); }
    std::unique_ptr<torch::optim::OptimizerParamState> AggMo::make_param_state() { return std::make_unique<AggMoParamState>(); }
}