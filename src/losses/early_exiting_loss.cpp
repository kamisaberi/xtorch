#include <losses/early_exiting_loss.h>

namespace xt::losses
{
    torch::Tensor early_exiting_loss(const std::vector<torch::Tensor>& exit_logits, const torch::Tensor& labels,
                                     const std::vector<float>& weights, float consistency_weight = 0.1f)
    {
        // Ensure inputs are valid
        TORCH_CHECK(!exit_logits.empty(), "Exit logits list must not be empty");
        TORCH_CHECK(labels.dim() == 1, "Labels must be 1D (batch_size)");
        TORCH_CHECK(labels.dtype() == torch::kLong, "Labels must be long type");
        TORCH_CHECK(weights.size() == exit_logits.size(), "Number of weights must match number of exits");

        int64_t batch_size = labels.size(0);
        int64_t num_classes = exit_logits[0].size(1);

        for (size_t i = 0; i < exit_logits.size(); ++i)
        {
            TORCH_CHECK(exit_logits[i].dim() == 2, "Exit logits must be 2D (batch_size, num_classes)");
            TORCH_CHECK(exit_logits[i].size(0) == batch_size, "Batch size mismatch in exit logits");
            TORCH_CHECK(exit_logits[i].size(1) == num_classes, "Number of classes mismatch in exit logits");
            TORCH_CHECK(exit_logits[i].dtype() == torch::kFloat, "Exit logits must be float type");
            TORCH_CHECK(weights[i] >= 0.0f, "Weights must be non-negative");
        }
        TORCH_CHECK(consistency_weight >= 0.0f, "Consistency weight must be non-negative");

        // Compute cross-entropy loss for each exit
        torch::Tensor total_loss = torch::zeros({}, torch::kFloat);
        for (size_t i = 0; i < exit_logits.size(); ++i)
        {
            auto ce_loss = torch::nn::functional::cross_entropy(exit_logits[i], labels);
            total_loss += weights[i] * ce_loss;
        }

        // Compute consistency loss (KL divergence between exit predictions)
        if (consistency_weight > 0.0f && exit_logits.size() > 1)
        {
            torch::Tensor consistency_loss = torch::zeros({}, torch::kFloat);
            for (size_t i = 1; i < exit_logits.size(); ++i)
            {
                auto prob_i = torch::softmax(exit_logits[i], 1);
                auto prob_0 = torch::softmax(exit_logits[0], 1).detach(); // Reference exit (first exit)
                auto kl_div = torch::mean(
                    torch::sum(prob_0 * (torch::log(prob_0 + 1e-10f) - torch::log(prob_i + 1e-10f)), 1));
                consistency_loss += kl_div;
            }
            consistency_loss /= (exit_logits.size() - 1); // Average over pairs
            total_loss += consistency_weight * consistency_loss;
        }

        return total_loss;
    }

    auto EarlyExitingLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        const std::vector<torch::Tensor>& exit_logits = {torch::zeros(10)};
        const std::vector<float>& weights = {0.2f};
        return xt::losses::early_exiting_loss(exit_logits, torch::zeros(10), weights);
    }
}
