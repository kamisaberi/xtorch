#include "include/optimizations/mas.h"
#include <stdexcept>
namespace xt::optim
{
    // --- MASOptions Methods ---
    void MASOptions::serialize(torch::serialize::OutputArchive& archive) const
    {
        archive.write("lr", this->lr());
        archive.write("beta1", beta1());
        archive.write("beta2", beta2());
        archive.write("eps", eps());
        archive.write("lambda", lambda());
    }

    void MASOptions::deserialize(torch::serialize::InputArchive& archive)
    {
        c10::IValue ivalue;
        if (archive.try_read("lr", ivalue)) { this->lr(ivalue.toDouble()); }
        if (archive.try_read("beta1", ivalue)) { beta1_ = ivalue.toDouble(); }
        if (archive.try_read("beta2", ivalue)) { beta2_ = ivalue.toDouble(); }
        if (archive.try_read("eps", ivalue)) { eps_ = ivalue.toDouble(); }
        if (archive.try_read("lambda", ivalue)) { lambda_ = ivalue.toDouble(); }
    }

    std::unique_ptr<torch::optim::OptimizerOptions> MASOptions::clone() const
    {
        auto cloned = std::make_unique<MASOptions>(this->lr());
        cloned->beta1(beta1()).beta2(beta2()).eps(eps()).lambda(lambda());
        return cloned;
    }

    // --- MASParamState Methods ---
    void MASParamState::serialize(torch::serialize::OutputArchive& archive) const
    {
        archive.write("step", step(), true);
        if (importance().defined()) archive.write("importance", importance(), true);
        if (optimal_param().defined()) archive.write("optimal_param", optimal_param(), true);
        if (exp_avg().defined()) archive.write("exp_avg", exp_avg(), true);
        if (exp_avg_sq().defined()) archive.write("exp_avg_sq", exp_avg_sq(), true);
    }

    void MASParamState::deserialize(torch::serialize::InputArchive& archive)
    {
        at::Tensor temp;
        if (archive.try_read("step", temp, true)) { step_ = temp; }
        if (archive.try_read("importance", temp, true)) { importance_ = temp; }
        if (archive.try_read("optimal_param", temp, true)) { optimal_param_ = temp; }
        if (archive.try_read("exp_avg", temp, true)) { exp_avg_ = temp; }
        if (archive.try_read("exp_avg_sq", temp, true)) { exp_avg_sq_ = temp; }
    }

    std::unique_ptr<torch::optim::OptimizerParamState> MASParamState::clone() const
    {
        auto cloned = std::make_unique<MASParamState>();
        if (step().defined()) cloned->step(step().clone());
        if (importance().defined()) cloned->importance(importance().clone());
        if (optimal_param().defined()) cloned->optimal_param(optimal_param().clone());
        if (exp_avg().defined()) cloned->exp_avg(exp_avg().clone());
        if (exp_avg_sq().defined()) cloned->exp_avg_sq(exp_avg_sq().clone());
        return cloned;
    }

    // --- MAS Implementation ---
    MAS::MAS(std::vector<torch::Tensor> params, MASOptions options)
        : torch::optim::Optimizer(std::move(params), std::make_unique<MASOptions>(options))
    {
    }

    torch::Tensor MAS::step(LossClosure closure)
    {
        torch::NoGradGuard no_grad;
        torch::Tensor loss = {};
        if (closure) { loss = closure(); }

        auto& group_options = static_cast<MASOptions&>(param_groups_[0].options());

        for (auto& p : param_groups_[0].params())
        {
            if (!p.grad().defined()) { continue; }

            auto grad = p.grad().clone(); // Clone to avoid modifying the original grad
            auto& state = static_cast<MASParamState&>(*state_.at(p.unsafeGetTensorImpl()));

            if (!state.step().defined())
            {
                state.step(torch::tensor(0.0, torch::dtype(torch::kFloat64).device(torch::kCPU)));
                state.importance(torch::zeros_like(p));
                state.optimal_param(p.detach().clone());
                state.exp_avg(torch::zeros_like(p));
                state.exp_avg_sq(torch::zeros_like(p));
            }
            state.step(state.step() + 1.0);
            double current_step_val = state.step().item<double>();

            // 1. Calculate and add the MAS regularization gradient
            if (state.importance().sum().item<float>() > 0)
            {
                // Only apply if importance has been computed
                auto& omega = state.importance();
                auto& optimal_p = state.optimal_param();
                auto mas_grad = group_options.lambda() * omega * (p.detach() - optimal_p);
                grad.add_(mas_grad);
            }

            // 2. Perform a standard Adam update with the modified gradient
            auto& m = state.exp_avg();
            auto& v = state.exp_avg_sq();
            double bias_correction1 = 1.0 - std::pow(group_options.beta1(), current_step_val);
            double bias_correction2 = 1.0 - std::pow(group_options.beta2(), current_step_val);

            m.mul_(group_options.beta1()).add_(grad, 1.0 - group_options.beta1());
            v.mul_(group_options.beta2()).addcmul_(grad, grad, 1.0 - group_options.beta2());

            auto m_hat = m / bias_correction1;
            auto v_hat = v / bias_correction2;
            auto denom = v_hat.sqrt().add_(group_options.eps());

            p.data().addcdiv_(m_hat, denom, -group_options.lr());
        }
        return loss;
    }

    // Special method to update importance weights after a task
    void MAS::compute_and_update_importance(xt::Module& model, xt::dataloaders::ExtendedDataLoader& data_loader)
    {
        torch::GradMode::set_enabled(true); // Ensure gradients can be computed
        model.train(false); // Set model to eval mode for stable forward passes

        // Accumulate importance gradients
        std::vector<torch::Tensor> importance_grads;
        for (const auto& p : model.parameters())
        {
            importance_grads.push_back(torch::zeros_like(p));
        }

        for (auto& batch : data_loader)
        {
            model.zero_grad();
            auto output = std::any_cast<torch::Tensor>(model.forward({batch.first}));

            // Compute squared L2 norm of the output (function output)
            auto l2_squared = torch::norm(output, 2).pow(2);

            // Backpropagate this value to get the gradients needed for Omega
            l2_squared.backward();

            for (size_t i = 0; i < model.parameters().size(); ++i)
            {
                importance_grads[i].add_(model.parameters()[i].grad().abs());
            }
        }

        // Update the state for each parameter
        for (size_t i = 0; i < model.parameters().size(); ++i)
        {
            auto& p = model.parameters()[i];
            auto& state = static_cast<MASParamState&>(*state_.at(p.unsafeGetTensorImpl()));

            // Update Omega: The new importance is the average of the absolute gradients
            auto new_importance = importance_grads[i] / data_loader.size();

            // Accumulate importance from previous tasks (optional, depends on MAS variant)
            // Here we just set the new importance. For multi-task, you might add.
            // state.importance(new_importance);

            // Update theta*: Store the current parameters as the "optimal" ones for this task
            state.optimal_param(p.detach().clone());

            // Optional: Reset Adam state for the new task
            state.step(torch::tensor(0.0, torch::dtype(torch::kFloat64).device(torch::kCPU)));
            state.exp_avg(torch::zeros_like(p));
            state.exp_avg_sq(torch::zeros_like(p));

        }

        model.train(true); // Set model back to train mode
        torch::GradMode::set_enabled(false);
    }


    // --- Boilerplate Methods ---
    void MAS::save(torch::serialize::OutputArchive& archive) const { torch::optim::Optimizer::save(archive); }
    void MAS::load(torch::serialize::InputArchive& archive) { torch::optim::Optimizer::load(archive); }
    std::unique_ptr<torch::optim::OptimizerParamState> MAS::make_param_state() { return std::make_unique<MASParamState>(); }
}