#ifndef NT_ASGD_OPTIMIZER_HPP
#define NT_ASGD_OPTIMIZER_HPP

#include <torch/torch.h>
#include <vector>
#include <memory>
#include <deque>

// --- Options for NT-ASGD Optimizer ---
struct NTASGDOptions : torch::optim::OptimizerOptions {
    explicit NTASGDOptions(double learning_rate = 0.01)
        : torch::optim::OptimizerOptions() {
        this->lr(learning_rate);
    }

    // Standard SGD parameters
    TORCH_ARG(double, weight_decay) = 1e-4;
    TORCH_ARG(double, lr) = 1e-6;

    // NT-ASGD specific parameters
    TORCH_ARG(long, n) = 5; // Window size for non-monotonic trigger condition

    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<torch::optim::OptimizerOptions> clone() const override;
};

// --- Parameter State for NT-ASGD ---
struct NTASGDParamState : torch::optim::OptimizerParamState {
    TORCH_ARG(torch::Tensor, step);
    TORCH_ARG(torch::Tensor, averaged_param); // The running average of the parameters (mu)
    TORCH_ARG(torch::Tensor, trigger_step);   // The step 'n' at which averaging was triggered

    // NTASGDParamState() = default;
    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<OptimizerParamState> clone() const override;
};

// --- NTASGD Optimizer Class ---
class NTASGD : public torch::optim::Optimizer {
public:
    NTASGD(std::vector<torch::Tensor> params, NTASGDOptions options);

    // The step function is non-standard: it needs the current loss to check the trigger
    torch::Tensor step(double current_loss, LossClosure closure = nullptr);

    // A standard step for cases where you don't have the loss (e.g., during warmup)
    torch::Tensor step(LossClosure closure = nullptr) override;

    // After training, swap the model's parameters with the averaged ones
    void swap_with_averaged_params();

    void save(torch::serialize::OutputArchive& archive) const override;
    void load(torch::serialize::InputArchive& archive) override;

protected:
    std::unique_ptr<torch::optim::OptimizerParamState> make_param_state() ;

private:
    std::deque<double> loss_window_; // Stores the recent losses to check the trigger
    bool is_triggered_ = false;
};

#endif // NT_ASGD_OPTIMIZER_HPP