#include "include/optimizations/nt_asgd.h"
#include <stdexcept>
#include <algorithm> // For std::min_element

namespace xt::optim
{
    // --- NTASGDOptions Methods ---
    void NTASGDOptions::serialize(torch::serialize::OutputArchive& archive) const {
        archive.write("lr", this->lr());
        archive.write("weight_decay", weight_decay());
        archive.write("n", n());
    }
    void NTASGDOptions::deserialize(torch::serialize::InputArchive& archive) {
        c10::IValue ivalue;
        if (archive.try_read("lr", ivalue)) { this->lr(ivalue.toDouble()); }
        if (archive.try_read("weight_decay", ivalue)) { weight_decay_ = ivalue.toDouble(); }
        if (archive.try_read("n", ivalue)) { n_ = ivalue.toInt(); }
    }
    std::unique_ptr<torch::optim::OptimizerOptions> NTASGDOptions::clone() const {
        auto cloned = std::make_unique<NTASGDOptions>(this->lr());
        cloned->weight_decay(weight_decay()).n(n());
        return cloned;
    }

    // --- NTASGDParamState Methods ---
    void NTASGDParamState::serialize(torch::serialize::OutputArchive& archive) const {
        archive.write("step", step(), true);
        if(averaged_param().defined()) archive.write("averaged_param", averaged_param(), true);
        if(trigger_step().defined()) archive.write("trigger_step", trigger_step(), true);
    }
    void NTASGDParamState::deserialize(torch::serialize::InputArchive& archive) {
        at::Tensor temp;
        if(archive.try_read("step", temp, true)) { step_ = temp; }
        if(archive.try_read("averaged_param", temp, true)) { averaged_param_ = temp; }
        if(archive.try_read("trigger_step", temp, true)) { trigger_step_ = temp; }
    }
    std::unique_ptr<torch::optim::OptimizerParamState> NTASGDParamState::clone() const {
        auto cloned = std::make_unique<NTASGDParamState>();
        if(step().defined()) cloned->step(step().clone());
        if(averaged_param().defined()) cloned->averaged_param(averaged_param().clone());
        if(trigger_step().defined()) cloned->trigger_step(trigger_step().clone());
        return cloned;
    }


    // --- NTASGD Implementation ---
    NTASGD::NTASGD(std::vector<torch::Tensor> params, NTASGDOptions options)
        : torch::optim::Optimizer(std::move(params), std::make_unique<NTASGDOptions>(options)) {}

    torch::Tensor NTASGD::step(double current_loss, LossClosure closure) {
        // Check the non-monotonic trigger condition
        auto& group_options = static_cast<NTASGDOptions&>(param_groups_[0].options());
        if (!is_triggered_) {
            if (!loss_window_.empty()) {
                double min_loss_in_window = *std::min_element(loss_window_.begin(), loss_window_.end());
                if (current_loss > min_loss_in_window) {
                    is_triggered_ = true;
                    std::cout << "NT-ASGD triggered at step " << (loss_window_.size()) << "!" << std::endl;
                    // Set the trigger step for all parameters
                    for (auto& group : param_groups_) {
                        for (auto& p : group.params()) {
                            auto& state = static_cast<NTASGDParamState&>(*state_.at(p.unsafeGetTensorImpl()));
                            state.trigger_step(state.step().clone());
                        }
                    }
                }
            }
            // Update loss window
            loss_window_.push_back(current_loss);
            if (loss_window_.size() > group_options.n()) {
                loss_window_.pop_front();
            }
        }

        // Call the standard step function to perform the update
        return this->step(closure);
    }

    torch::Tensor NTASGD::step(LossClosure closure) {
        torch::NoGradGuard no_grad;
        torch::Tensor loss = {};
        if (closure) { loss = closure(); }

        auto& group_options = static_cast<NTASGDOptions&>(param_groups_[0].options());

        for (auto& group : param_groups_) {
            for (auto& p : group.params()) {
                if (!p.grad().defined()) { continue; }

                auto grad = p.grad();
                auto& state = static_cast<NTASGDParamState&>(*state_.at(p.unsafeGetTensorImpl()));

                if (!state.step().defined()) {
                    state.step(torch::tensor(0.0, torch::dtype(torch::kFloat64).device(torch::kCPU)));
                    state.averaged_param(p.detach().clone()); // Start average with initial param
                    state.trigger_step(torch::tensor(-1.0)); // -1 indicates not triggered yet
                }
                state.step(state.step() + 1.0);

                // 1. Standard SGD update
                if (group_options.weight_decay() > 0.0) {
                    grad.add_(p.detach(), group_options.weight_decay());
                }
                p.data().add_(grad, -group_options.lr());

                // 2. Update the averaged parameters if triggered
                if (state.trigger_step().item<double>() >= 0) {
                    double t = state.step().item<double>();
                    double n = state.trigger_step().item<double>();
                    double eta = 1.0 / (t - n + 1.0);

                    auto& mu = state.averaged_param();
                    mu.mul_(1.0 - eta).add_(p.detach(), eta);
                }
            }
        }
        return loss;
    }

    void NTASGD::swap_with_averaged_params() {
        torch::NoGradGuard no_grad;
        for (auto& group : param_groups_) {
            for (auto& p : group.params()) {
                auto& state = static_cast<NTASGDParamState&>(*state_.at(p.unsafeGetTensorImpl()));
                if (state.trigger_step().item<double>() >= 0) {
                    // Swap the model parameter with the averaged parameter
                    // A simple copy is sufficient here.
                    p.data().copy_(state.averaged_param());
                }
            }
        }
    }

    // --- Boilerplate Methods ---
    void NTASGD::save(torch::serialize::OutputArchive& archive) const { torch::optim::Optimizer::save(archive); }
    void NTASGD::load(torch::serialize::InputArchive& archive) { torch::optim::Optimizer::load(archive); }
    std::unique_ptr<torch::optim::OptimizerParamState> NTASGD::make_param_state() { return std::make_unique<NTASGDParamState>(); }
}