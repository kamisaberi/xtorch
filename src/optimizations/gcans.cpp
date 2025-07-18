#include <optimizations/gcans.h>
#include <stdexcept>

namespace xt::optim
{
    // --- GCANSOptions Methods ---
    void GCANSOptions::serialize(torch::serialize::OutputArchive& archive) const
    {
        archive.write("lr", this->lr);
        archive.write("momentum", momentum());
        archive.write("weight_decay", weight_decay());
        archive.write("compression_ratio", compression_ratio());
        archive.write("sampling_beta", sampling_beta());
    }

    void GCANSOptions::deserialize(torch::serialize::InputArchive& archive)
    {
        c10::IValue ivalue;
        if (archive.try_read("lr", ivalue)) { this->lr = ivalue.toDouble(); }
        if (archive.try_read("momentum", ivalue)) { momentum_ = ivalue.toDouble(); }
        if (archive.try_read("weight_decay", ivalue)) { weight_decay_ = ivalue.toDouble(); }
        if (archive.try_read("compression_ratio", ivalue)) { compression_ratio_ = ivalue.toDouble(); }
        if (archive.try_read("sampling_beta", ivalue)) { sampling_beta_ = ivalue.toDouble(); }
    }

    std::unique_ptr<torch::optim::OptimizerOptions> GCANSOptions::clone() const
    {
        auto cloned = std::make_unique<GCANSOptions>(this->lr);
        cloned->momentum(momentum()).weight_decay(weight_decay())
              .compression_ratio(compression_ratio()).sampling_beta(sampling_beta());
        return cloned;
    }

    // --- GCANSParamState Methods ---
    void GCANSParamState::serialize(torch::serialize::OutputArchive& archive) const
    {
        archive.write("step", step(), true);
        if (momentum_buffer().defined()) archive.write("momentum_buffer", momentum_buffer(), true);
        if (error_feedback().defined()) archive.write("error_feedback", error_feedback(), true);
        if (sampling_probs().defined()) archive.write("sampling_probs", sampling_probs(), true);
    }

    void GCANSParamState::deserialize(torch::serialize::InputArchive& archive)
    {
        at::Tensor temp;
        if (archive.try_read("step", temp, true)) { step_ = temp; }
        if (archive.try_read("momentum_buffer", temp, true)) { momentum_buffer_ = temp; }
        if (archive.try_read("error_feedback", temp, true)) { error_feedback_ = temp; }
        if (archive.try_read("sampling_probs", temp, true)) { sampling_probs_ = temp; }
    }

    std::unique_ptr<torch::optim::OptimizerParamState> GCANSParamState::clone() const
    {
        auto cloned = std::make_unique<GCANSParamState>();
        if (step().defined()) cloned->step(step().clone());
        if (momentum_buffer().defined()) cloned->momentum_buffer(momentum_buffer().clone());
        if (error_feedback().defined()) cloned->error_feedback(error_feedback().clone());
        if (sampling_probs().defined()) cloned->sampling_probs(sampling_probs().clone());
        return cloned;
    }

    // --- GCANS Implementation ---
    GCANS::GCANS(std::vector<torch::Tensor> params, GCANSOptions options)
        : torch::optim::Optimizer(std::move(params), std::make_unique<GCANSOptions>(options))
    {
    }

    torch::Tensor GCANS::step(LossClosure closure)
    {
        torch::NoGradGuard no_grad;
        torch::Tensor loss = {};
        if (closure) { loss = closure(); }

        auto& group_options = static_cast<GCANSOptions&>(param_groups_[0].options());

        for (auto& p : param_groups_[0].params())
        {
            if (!p.grad().defined()) { continue; }

            auto grad = p.grad();
            auto& state = static_cast<GCANSParamState&>(*state_.at(p.unsafeGetTensorImpl()));

            if (!state.step().defined())
            {
                state.step(torch::tensor(0.0, torch::dtype(torch::kFloat64).device(torch::kCPU)));
                state.momentum_buffer(torch::zeros_like(p));
                state.error_feedback(torch::zeros_like(p));
                // Initialize sampling probabilities uniformly
                state.sampling_probs(torch::ones_like(p) / p.numel());
            }
            state.step(state.step() + 1.0);

            // 1. Compensate gradient with local error feedback
            auto compensated_grad = grad + state.error_feedback();

            // 2. Update adaptive sampling probabilities
            // P_t = beta * P_{t-1} + (1-beta) * |g_t|
            auto& sampling_probs = state.sampling_probs();
            sampling_probs.mul_(group_options.sampling_beta()).add_(compensated_grad.abs(),
                                                                    1.0 - group_options.sampling_beta());

            // Normalize probabilities to sum to 1
            auto probs_sum = sampling_probs.sum();
            if (probs_sum.item<double>() > 1e-10)
            {
                sampling_probs.div_(probs_sum);
            }

            // 3. Select Top-K gradient values to form the coreset
            // The selection score is |compensated_grad| / sampling_probs
            // A small epsilon is added to probs to avoid division by zero
            auto selection_scores = compensated_grad.abs() / (sampling_probs + 1e-12);

            int64_t k = std::max(1L, static_cast<int64_t>(p.numel() * group_options.compression_ratio()));

            // Find the k-th largest score to use as a threshold
            auto threshold = std::get<0>(torch::kthvalue(selection_scores.flatten(), selection_scores.numel() - k));

            // Create the sparse gradient mask
            auto mask = (selection_scores >= threshold).to(grad.dtype());

            // The sparse gradient to be "communicated" and used for the update
            auto sparse_grad = compensated_grad * mask;

            // 4. Update local error feedback buffer
            // The new error is the part of the gradient we did *not* select
            state.error_feedback(compensated_grad - sparse_grad);

            // 5. Apply the base optimizer (SGD with Momentum) using the sparse gradient
            if (group_options.weight_decay() > 0.0)
            {
                sparse_grad.add_(p.detach(), group_options.weight_decay());
            }

            auto& momentum_buffer = state.momentum_buffer();
            momentum_buffer.mul_(group_options.momentum()).add_(sparse_grad);

            p.data().add_(momentum_buffer, -group_options.lr);
        }
        return loss;
    }

    // --- Boilerplate Methods ---
    void GCANS::save(torch::serialize::OutputArchive& archive) const { torch::optim::Optimizer::save(archive); }
    void GCANS::load(torch::serialize::InputArchive& archive) { torch::optim::Optimizer::load(archive); }

    std::unique_ptr<torch::optim::OptimizerParamState> GCANS::make_param_state()
    {
        return std::make_unique<GCANSParamState>();
    }
}