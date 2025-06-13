#ifndef SMA_OPTIMIZER_HPP
#define SMA_OPTIMIZER_HPP

#include <torch/torch.h>
#include <torch/serialize/archive.h>

#include <cmath>
#include <vector>
#include <memory>
#include <string>
#include <cstdint>
#include <deque>

// --- Options for SMA Optimizer ---
struct SMAOptions : torch::optim::OptimizerOptions {
    explicit SMAOptions(double learning_rate = 1e-3)
        : torch::optim::OptimizerOptions() {
        this->lr(learning_rate);
    }

    // Window size for the Simple Moving Averages
    TORCH_ARG(long, window_size) = 10;

    TORCH_ARG(double, eps) = 1e-8;
    TORCH_ARG(double, weight_decay) = 0.0;
    TORCH_ARG(double, lr) = 1e-6;

    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<torch::optim::OptimizerOptions> clone() const override;
};

// --- Parameter State for SMA Optimizer ---
struct SMAParamState : torch::optim::OptimizerParamState {
    // We don't use TORCH_ARG for the deque as it's not a simple type.
    // Serialization will be handled manually.
    std::deque<torch::Tensor> grad_window;
    std::deque<torch::Tensor> grad_sq_window;

    // We can still store sums to avoid re-computing them every time.
    TORCH_ARG(torch::Tensor, grad_sum);
    TORCH_ARG(torch::Tensor, grad_sq_sum);

    // SMAParamState() = default;
    void serialize(torch::serialize::OutputArchive& archive) const override;
    void deserialize(torch::serialize::InputArchive& archive) ;
    std::unique_ptr<OptimizerParamState> clone() const override;
};

// --- SMA Optimizer Class ---
class SMA : public torch::optim::Optimizer {
public:
    SMA(std::vector<torch::Tensor> params, SMAOptions options);
    explicit SMA(std::vector<torch::Tensor> params, double lr = 1e-3);

    using LossClosure = std::function<torch::Tensor()>;
    torch::Tensor step(LossClosure closure = nullptr) override;
    void save(torch::serialize::OutputArchive& archive) const override;
    void load(torch::serialize::InputArchive& archive) override;

protected:
    std::unique_ptr<torch::optim::OptimizerParamState> make_param_state() ;
};

#endif // SMA_OPTIMIZER_HPP