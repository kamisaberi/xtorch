#include <optimizations/1_bit_adam.h> // Your header file
#include <stdexcept>       // For std::runtime_error
#include <cmath>           // For std::pow, std::sqrt

// --- OneBitAdam Class Implementation ---
namespace xt::optim
{
    OneBitAdam::OneBitAdam(std::vector<torch::Tensor> params, OneBitAdamOptions options)
        : torch::optim::Optimizer(std::move(params), std::make_unique<OneBitAdamOptions>(options))
    {
    }

    OneBitAdam::OneBitAdam(std::vector<torch::Tensor> params, double lr_val)
        : OneBitAdam(std::move(params), OneBitAdamOptions(lr_val))
    {
    }

    torch::Tensor OneBitAdam::step(LossClosure closure)
    {
        torch::NoGradGuard no_grad;

        torch::Tensor loss = {};
        if (closure)
        {
            loss = closure();
        }

        for (auto& group : param_groups_)
        {
            auto& options = static_cast<OneBitAdamOptions&>(group.options());

            for (auto& p : group.params())
            {
                if (!p.grad().defined())
                {
                    continue;
                }

                auto grad = p.grad();
                if (grad.is_sparse())
                {
                    throw std::runtime_error("OneBitAdam does not support sparse gradients at this time.");
                }

                // --- Manual State Management ---
                // Access the state_ map directly. state_ is a protected member:
                // std::unordered_map<void*, std::unique_ptr<OptimizerParamState>> state_;
                // The key is p.unsafeGetTensorImpl().

                if (state_.find(p.unsafeGetTensorImpl()) == state_.end())
                {
                    // State for this parameter does not exist, create it.
                    // This is where OneBitAdam::make_param_state() *would* have been useful
                    // if the base class's state_for() was calling it.
                    // Since we are managing manually, we create it here.
                    state_[p.unsafeGetTensorImpl()] = std::make_unique<OneBitAdamParamState>();
                }

                // Now, get the reference to our custom state type.
                // This cast is safe IF state_ only ever contains OneBitAdamParamState for this optimizer.
                // This will be true if make_param_state override was working OR if load() is also customized.
                // If Optimizer::load() (without make_param_state override) ran, this cast will fail after loading.
                auto& state = static_cast<OneBitAdamParamState&>(*(state_[p.unsafeGetTensorImpl()]));
                // --- End Manual State Management ---

                if (!state.step().defined())
                {
                    state.step(torch::tensor(0.0, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)));
                    auto p_options = p.options();
                    state.exp_avg(torch::zeros_like(p, p_options.memory_format(torch::MemoryFormat::Preserve)));
                    state.exp_avg_sq(torch::zeros_like(p, p_options.memory_format(torch::MemoryFormat::Preserve)));
                    state.error_feedback(torch::zeros_like(p, p_options.memory_format(torch::MemoryFormat::Preserve)));
                    state.momentum_buffer(torch::zeros_like(p, p_options.memory_format(torch::MemoryFormat::Preserve)));
                }

                auto& exp_avg = state.exp_avg();
                auto& exp_avg_sq = state.exp_avg_sq();
                auto& error_feedback = state.error_feedback();
                auto& momentum_buffer = state.momentum_buffer();

                torch::Tensor current_step_tensor = state.step();
                current_step_tensor.add_(1.0);
                state.step(current_step_tensor);
                double current_step_val = current_step_tensor.item<double>();

                if (options.weight_decay() != 0.0)
                {
                    grad.add_(p.detach(), options.weight_decay());
                }

                double beta1 = options.beta1();
                double beta2 = options.beta2();

                exp_avg.mul_(beta1).add_(grad, 1.0 - beta1);
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, 1.0 - beta2);

                double bias_correction1 = 1.0 - std::pow(beta1, current_step_val);
                double bias_correction2 = 1.0 - std::pow(beta2, current_step_val);

                torch::Tensor m_t_hat_for_update;

                if (current_step_val <= options.warmup_steps())
                {
                    m_t_hat_for_update = exp_avg / bias_correction1;
                    momentum_buffer.copy_(m_t_hat_for_update);
                    error_feedback.zero_();
                }
                else
                {
                    auto m_t_compensated = (exp_avg / bias_correction1).add_(error_feedback);
                    auto scale = m_t_compensated.abs().mean();

                    if (torch::isnan(scale).any().item<bool>() || torch::isinf(scale).any().item<bool>() ||
                        (scale.item<double>() < 1e-10 && scale.item<double>() > -1e-10) )
                    {
                        m_t_hat_for_update = torch::sign(m_t_compensated);
                        if (std::abs(scale.item<double>()) < 1e-20) { // Check if scale is truly zero
                            m_t_hat_for_update.zero_();
                        }
                    }
                    else
                    {
                        m_t_hat_for_update = torch::sign(m_t_compensated) * scale;
                    }
                    error_feedback.copy_(m_t_compensated - m_t_hat_for_update);
                }

                auto denom = (exp_avg_sq / bias_correction2).sqrt_().add_(options.eps());
                p.data().addcdiv_(m_t_hat_for_update, denom, -options.lr);
            }
        }
        return loss;
    }

    void OneBitAdam::save(torch::serialize::OutputArchive& archive) const
    {
        // This will call OneBitAdamParamState::serialize for each state object, which is correct.
        torch::optim::Optimizer::save(archive);
    }

    void OneBitAdam::load(torch::serialize::InputArchive& archive)
    {
        // WARNING: If your LibTorch version's Optimizer::load() does not use an overridden
        // make_param_state() (because the override isn't working or the base doesn't have it as virtual),
        // then this base load() will populate `state_` with base `OptimizerParamState` objects.
        // The static_cast in `step()` will then fail at runtime after loading.
        // To fix this, you would need to implement a fully custom `OneBitAdam::load` method
        // that manually creates `OneBitAdamParamState` instances and calls their `deserialize` method.
        // This is a non-trivial task.
        torch::optim::Optimizer::load(archive);

        // After base load, you might need to iterate through state_ and verify/recast,
        // or implement fully custom loading. For example (conceptual, might need refinement):
        //
        // torch::optim::Optimizer::load_param_groups(archive); // Load param groups and options
        //
        // // The following is pseudo-code for how you might approach custom state loading
        // // if the base Optimizer::load() is problematic for custom states.
        // // This part is highly dependent on how Optimizer::save() structures the archive.
        // c10::Dict<std::string, torch::Tensor> flat_state_dict; // Or appropriate dict type
        // if (archive.try_read("state", flat_state_dict)) { // Check key "state"
        //     // Reconstruct state_ map
        //     state_.clear(); // Clear any states created by base (if any)
        //     for (auto& group : param_groups_) {
        //         for (auto& p : group.params()) {
        //             // Need a way to map param 'p' to its serialized ID used in flat_state_dict.
        //             // This is where it gets complex as Optimizer's internal param IDing isn't public.
        //             // Assuming you can get the serialized state for param 'p':
        //             //
        //             // auto param_state_for_p_archive_view = ... get data from flat_state_dict ...
        //             // auto new_state = std::make_unique<OneBitAdamParamState>();
        //             // new_state->deserialize(param_state_for_p_archive_view_as_InputArchive); // Needs careful handling
        //             // state_[p.unsafeGetTensorImpl()] = std::move(new_state);
        //         }
        //     }
        // } else {
        //     TORCH_WARN("Could not read 'state' from archive during OneBitAdam::load");
        // }
    }

    // Factory method for creating parameter-specific states.
    // If state_for() and the override mechanism for make_param_state() were working,
    // the base class would call this. With manual state_ map management in step(),
    // this method as defined in OneBitAdam.cpp might not be called by the base Optimizer's
    // typical pathways (like its own state_for() or its load()).
    // If you implement a fully custom load(), you might call this method yourself.
    // For now, if it's causing "doesn't exist" linker errors or override errors,
    // it might be best to remove its definition if it's not being successfully used.
    // If the header declares it (e.g., as a protected override attempt), you need a definition.
    // I'm keeping the definition here assuming the header still declares it.
    /*
    std::unique_ptr<torch::optim::OptimizerParamState> OneBitAdam::make_param_state()
    {
        // This function's utility is diminished if the base Optimizer's state creation/loading
        // mechanisms are not polymorphically calling it due to version/override issues.
        return std::make_unique<OneBitAdamParamState>();
    }
    */
    // If the above make_param_state causes linker errors ("undefined reference") and you are
    // sure its declaration in the header is problematic (e.g. `override` fails),
    // you might have removed the declaration from the header. If so, remove this definition too.
    // If the header still has the declaration, you NEED this definition to link.
    // For now, I'm commenting it out, assuming the "doesn't exist" might also refer to
    // this not being properly linked or its declaration being problematic.
    // If your header still has `make_param_state` declared, you MUST uncomment this definition.
}