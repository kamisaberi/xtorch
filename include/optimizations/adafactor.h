#ifndef ADAFACTOR_OPTIMIZER_HPP
#define ADAFACTOR_OPTIMIZER_HPP

#include <torch/torch.h>
#include <torch/serialize/archive.h>

#include <cmath>
#include <vector>
#include <memory>
#include <string>
#include <cstdint>

// --- Options for Adafactor Optimizer ---
struct AdafactorOptions : torch::optim::OptimizerOptions {
    explicit AdafactorOptions(c10::optional<double> learning_rate = c10::nullopt)
        : torch::optim::OptimizerOptions() {
        if (learning_rate.has_value()) {
            this->lr(learning_rate.value());
        }
    }

    // beta2 is used for the EMA of the factorized second moments
    TORCH_ARG(double, beta2) = 0.999;
    TORCH_ARG(double, eps1) = 1e-30; // Small constant for regularizing squared gradients
    TORCH_ARG(double, eps2) = 1e-3;  // Small constant for regularizing the final update
    TORCH_ARG(double, clip_threshold) = 1.0;
    TORCH_ARG(double, weight_decay) = 0.0;
    TORCH_ARG(bool, scale_parameter) = true; // Whether to use relative step size scaling
    TORCH_ARG(bool, relative_step) = true; // Use relative step size or fixed LR

    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) override;
    std::unique_ptr<torch::optim::OptimizerOptions> clone() const override;
};

// --- Parameter State for Adafactor Optimizer ---
struct AdafactorParamState : torch::optim::OptimizerParamState {
    TORCH_ARG(torch::Tensor, step);
public:

    // Factorized second moment estimates
    torch::Tensor exp_avg_sq_row; // R_t
    torch::Tensor exp_avg_sq_col; // C_t

    // Note: No exp_avg (m_t) is stored in the default memory-efficient version.

    AdafactorParamState() = default;
    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) override;
    std::unique_ptr<OptimizerParamState> clone() const override;
};

// --- Adafactor Optimizer Class ---
class Adafactor : public torch::optim::Optimizer {
public:
    Adafactor(std::vector<torch::Tensor> params, AdafactorOptions options);

    using LossClosure = std::function<torch::Tensor()>;
    torch::Tensor step(LossClosure closure = nullptr) override;
    void save(torch::serialize::OutputArchive& archive) const override;
    void load(torch::serialize::InputArchive& archive) override;

protected:
    std::unique_ptr<torch::optim::OptimizerParamState> make_param_state() override;

private:
    double _get_relative_step_size(const torch::Tensor& param, long step);
};

#endif // ADAFACTOR_OPTIMIZER_HPP