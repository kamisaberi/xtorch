#include "include/optimizations/deep_ensembles.h"
#include <stdexcept>

#include "base/module.h"
namespace xt::optim
{
    // --- DeepEnsemblesOptimizer Implementation ---

    DeepEnsemblesOptimizer::DeepEnsemblesOptimizer(
        std::function<std::shared_ptr<xt::Module>()> model_factory,
        DeepEnsemblesOptions options)
        : options_(std::move(options))
    {
        TORCH_CHECK(model_factory, "A valid model factory function must be provided.");
        TORCH_CHECK(options_.ensemble_size > 0, "Ensemble size must be positive.");

        // Create the ensemble of models and their corresponding optimizers
        for (int i = 0; i < options_.ensemble_size; ++i)
        {
            // Create a new, independently initialized model
            auto model = model_factory();
            ensemble_models_.push_back(model);

            // Create a dedicated AdamW optimizer for this model
            auto optimizer = std::make_unique<torch::optim::AdamW>(
                model->parameters(),
                options_.inner_optimizer_options
            );
            ensemble_optimizers_.push_back(std::move(optimizer));
        }
    }

    void DeepEnsemblesOptimizer::zero_grad()
    {
        for (auto& optimizer : ensemble_optimizers_)
        {
            optimizer->zero_grad();
        }
    }

    std::vector<torch::Tensor> DeepEnsemblesOptimizer::step(
        const torch::Tensor& input,
        const torch::Tensor& target,
        const std::function<torch::Tensor(torch::Tensor, torch::Tensor)>& loss_fn)
    {
        std::vector<torch::Tensor> losses;

        // Iterate through the ensemble, training each model independently on the batch
        for (int i = 0; i < options_.ensemble_size; ++i)
        {
            auto& model = ensemble_models_[i];
            auto& optimizer = ensemble_optimizers_[i];

            // Forward pass for this specific model
            auto output = std::any_cast<torch::Tensor>(model->forward({input}));

            // Compute loss
            auto loss = loss_fn(output, target);
            losses.push_back(loss.detach().clone());

            // Backward pass to compute gradients for this model
            loss.backward();

            // Update this model's weights using its dedicated optimizer
            optimizer->step();
        }

        return losses;
    }

    torch::Tensor DeepEnsemblesOptimizer::predict(const torch::Tensor& input)
    {
        torch::NoGradGuard no_grad;
        std::vector<torch::Tensor> all_predictions;

        // Get predictions from every model in the ensemble
        for (const auto& model : ensemble_models_)
        {
            auto output = std::any_cast<torch::Tensor>(model->forward({input}));
            // For classification, it's common to average probabilities (after softmax)
            // For regression, you'd average the raw outputs.
            // We'll assume classification and apply softmax.
            auto probabilities = torch::softmax(output, /*dim=*/1);
            all_predictions.push_back(probabilities.unsqueeze(0)); // Add a dimension for stacking
        }

        // Stack predictions and compute the mean across the ensemble dimension
        auto stacked_preds = torch::cat(all_predictions, 0);
        auto mean_preds = torch::mean(stacked_preds, 0);

        return mean_preds;
    }

    void DeepEnsemblesOptimizer::train()
    {
        for (auto& model : ensemble_models_)
        {
            model->train();
        }
    }

    void DeepEnsemblesOptimizer::eval()
    {
        for (auto& model : ensemble_models_)
        {
            model->eval();
        }
    }
}