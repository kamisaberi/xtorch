#include "include/optimizations/gradient_sparsification.h"
#include <stdexcept>

namespace xt::optim
{
    // --- GradientSparsificationOptions Methods ---
    void GradientSparsificationOptions::serialize(torch::serialize::OutputArchive& archive) const {
        archive.write("lr", this->lr);
        archive.write("momentum", momentum());
        archive.write("weight_decay", weight_decay());
        archive.write("compression_ratio", compression_ratio());
    }

    void GradientSparsificationOptions::deserialize(torch::serialize::InputArchive& archive) {
        c10::IValue ivalue;
        if (archive.try_read("lr", ivalue)) { this->lr=ivalue.toDouble(); }
        if (archive.try_read("momentum", ivalue)) { momentum_ = ivalue.toDouble(); }
        if (archive.try_read("weight_decay", ivalue)) { weight_decay_ = ivalue.toDouble(); }
        if (archive.try_read("compression_ratio", ivalue)) { compression_ratio_ = ivalue.toDouble(); }
    }

    std::unique_ptr<torch::optim::OptimizerOptions> GradientSparsificationOptions::clone() const {
        auto cloned = std::make_unique<GradientSparsificationOptions>(this->lr);
        cloned->momentum(momentum()).weight_decay(weight_decay())
              .compression_ratio(compression_ratio());
        return cloned;
    }

    // --- GSParamState Methods ---
    void GSParamState::serialize(torch::serialize::OutputArchive& archive) const {
        archive.write("step", step(), true);
        if(momentum_buffer().defined()) archive.write("momentum_buffer", momentum_buffer(), true);
        if(error_feedback().defined()) archive.write("error_feedback", error_feedback(), true);
    }

    void GSParamState::deserialize(torch::serialize::InputArchive& archive) {
        at::Tensor temp;
        if(archive.try_read("step", temp, true)) { step_ = temp; }
        if(archive.try_read("momentum_buffer", temp, true)) { momentum_buffer_ = temp; }
        if(archive.try_read("error_feedback", temp, true)) { error_feedback_ = temp; }
    }

    std::unique_ptr<torch::optim::OptimizerParamState> GSParamState::clone() const {
        auto cloned = std::make_unique<GSParamState>();
        if(step().defined()) cloned->step(step().clone());
        if(momentum_buffer().defined()) cloned->momentum_buffer(momentum_buffer().clone());
        if(error_feedback().defined()) cloned->error_feedback(error_feedback().clone());
        return cloned;
    }

    // --- GradientSparsification Implementation ---
    GradientSparsification::GradientSparsification(std::vector<torch::Tensor> params, GradientSparsificationOptions options)
        : torch::optim::Optimizer(std::move(params), std::make_unique<GradientSparsificationOptions>(options)) {}

    torch::Tensor GradientSparsification::step(LossClosure closure) {
        torch::NoGradGuard no_grad;
        torch::Tensor loss = {};
        if (closure) { loss = closure(); }

        auto& group_options = static_cast<GradientSparsificationOptions&>(param_groups_[0].options());

        for (auto& p : param_groups_[0].params()) {
            if (!p.grad().defined()) { continue; }

            auto grad = p.grad();
            auto& state = static_cast<GSParamState&>(*state_.at(p.unsafeGetTensorImpl()));

            if (!state.step().defined()) {
                state.step(torch::tensor(0.0, torch::dtype(torch::kFloat64).device(torch::kCPU)));
                state.momentum_buffer(torch::zeros_like(p));
                state.error_feedback(torch::zeros_like(p));
            }
            state.step(state.step() + 1.0);

            // 1. Compensate gradient with local error feedback from previous steps
            auto compensated_grad = grad + state.error_feedback();

            // 2. Select Top-K gradient values
            // Determine k based on the compression ratio
            int64_t k = std::max(1L, static_cast<int64_t>(p.numel() * group_options.compression_ratio()));

            // Find the k-th largest absolute value in the compensated gradient
            auto topk_abs_values = torch::abs(compensated_grad).flatten();
            auto threshold_tensor = std::get<0>(torch::kthvalue(topk_abs_values, topk_abs_values.numel() - k));
            auto threshold = threshold_tensor.item<float>();

            // Create a binary mask for values >= threshold
            auto mask = (torch::abs(compensated_grad) >= threshold).to(grad.dtype());

            // Create the sparse gradient by applying the mask
            auto sparse_grad = compensated_grad * mask;

            // 3. Update the local error feedback buffer for the next iteration
            // The new error is the part of the gradient we did *not* select
            state.error_feedback(compensated_grad - sparse_grad);

            // 4. Apply the base optimizer (SGD with Momentum) using the selected sparse gradient
            // First, apply weight decay if any
            if (group_options.weight_decay() > 0.0) {
                // Note: weight decay is applied to the sparse gradient.
                // This is a design choice. Applying it to the dense gradient before
                // selection is another valid option.
                sparse_grad.add_(p.detach(), group_options.weight_decay());
            }

            // Update momentum buffer
            auto& momentum_buffer = state.momentum_buffer();
            momentum_buffer.mul_(group_options.momentum()).add_(sparse_grad);

            // Apply the final update to the parameters
            p.data().add_(momentum_buffer, -group_options.lr);
        }
        return loss;
    }

    // --- Boilerplate Methods ---
    void GradientSparsification::save(torch::serialize::OutputArchive& archive) const { torch::optim::Optimizer::save(archive); }
    void GradientSparsification::load(torch::serialize::InputArchive& archive) { torch::optim::Optimizer::load(archive); }
    std::unique_ptr<torch::optim::OptimizerParamState> GradientSparsification::make_param_state() { return std::make_unique<GSParamState>(); }
}