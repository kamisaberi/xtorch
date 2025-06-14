#ifndef DEEP_ENSEMBLES_OPTIMIZER_HPP
#define DEEP_ENSEMBLES_OPTIMIZER_HPP

#include <torch/torch.h>
#include <vector>
#include <memory>
#include <functional>

// Forward declaration of the model class is good practice
struct MyModel;

// --- Options for DeepEnsemblesOptimizer ---
struct DeepEnsemblesOptions {
    int ensemble_size = 5;
    // We will use standard AdamWOptions for each inner optimizer
    torch::optim::AdamWOptions inner_optimizer_options = torch::optim::AdamWOptions(1e-3);
};

// --- DeepEnsemblesOptimizer Class (Meta-Optimizer) ---
class DeepEnsemblesOptimizer {
public:
    // The constructor takes a factory function to create new model instances
    DeepEnsemblesOptimizer(
        std::function<std::shared_ptr<torch::nn::Module>()> model_factory,
        DeepEnsemblesOptions options);

    // The step function needs data and a loss function to train the ensemble
    std::vector<torch::Tensor> step(const torch::Tensor& input, const torch::Tensor& target,
                                    const std::function<torch::Tensor(torch::Tensor, torch::Tensor)>& loss_fn);

    // A convenience method for inference that averages predictions
    torch::Tensor predict(const torch::Tensor& input);

    void zero_grad();
    void train(); // Helper to set all models to train mode
    void eval();  // Helper to set all models to eval mode

private:
    std::vector<std::shared_ptr<torch::nn::Module>> ensemble_models_;
    std::vector<std::unique_ptr<torch::optim::AdamW>> ensemble_optimizers_;
    DeepEnsemblesOptions options_;
};

#endif // DEEP_ENSEMBLES_OPTIMIZER_HPP