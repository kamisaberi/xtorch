#include <losses/multi_loss.h>

namespace xt::losses
{
    torch::Tensor multi_loss(const std::vector<torch::Tensor>& predictions, const std::vector<torch::Tensor>& targets,
                             const std::vector<float>& weights,
                             const std::vector<bool>& is_classification,
                             float reg_weight = 0.1f
    )
    {
        // Ensure inputs are valid
        TORCH_CHECK(!predictions.empty(), "Predictions list must not be empty");
        TORCH_CHECK(predictions.size() == targets.size(), "Number of predictions must match number of targets");
        TORCH_CHECK(predictions.size() == weights.size(), "Number of weights must match number of predictions");
        TORCH_CHECK(predictions.size() == is_classification.size(),
                    "Number of is_classification flags must match number of predictions");
        TORCH_CHECK(reg_weight >= 0.0f, "Regularization weight must be non-negative");

        int64_t batch_size = predictions[0].size(0);
        torch::Tensor total_loss = torch::zeros({}, torch::kFloat);

        // Compute task-specific losses
        for (size_t i = 0; i < predictions.size(); ++i)
        {
            const auto& pred = predictions[i];
            const auto& target = targets[i];
            float weight = weights[i];

            TORCH_CHECK(pred.size(0) == batch_size, "Batch size mismatch in predictions");
            TORCH_CHECK(target.size(0) == batch_size, "Batch size mismatch in targets");
            TORCH_CHECK(pred.dtype() == torch::kFloat, "Predictions must be float type");
            TORCH_CHECK(weight >= 0.0f, "Weight must be non-negative");

            if (is_classification[i])
            {
                // Classification: Cross-entropy loss
                TORCH_CHECK(target.dtype() == torch::kLong, "Target must be long type for classification");
                TORCH_CHECK(pred.dim() == 2, "Classification predictions must be 2D (batch_size, num_classes)");
                TORCH_CHECK(target.dim() == 1, "Classification targets must be 1D (batch_size)");
                auto ce_loss = torch::nn::functional::cross_entropy(pred, target);
                total_loss += weight * ce_loss;
            }
            else
            {
                // Regression: Smooth L1 loss
                TORCH_CHECK(target.dtype() == torch::kFloat, "Target must be float type for regression");
                TORCH_CHECK(pred.sizes() == target.sizes(),
                            "Prediction and target must have the same shape for regression");
                auto diff = torch::abs(pred - target);
                auto beta = 1.0f; // Fixed beta for Smooth L1
                auto mask = diff < beta;
                auto loss_small = 0.5f * diff * diff / beta * mask;
                auto loss_large = (diff - 0.5f * beta) * (1.0f - mask.to(torch::kFloat));
                auto smooth_l1_loss = (loss_small + loss_large).mean();
                total_loss += weight * smooth_l1_loss;
            }
        }

        // Add L2 regularization on predictions (optional)
        if (reg_weight > 0.0f)
        {
            torch::Tensor reg_loss = torch::zeros({}, torch::kFloat);
            for (const auto& pred : predictions)
            {
                reg_loss += torch::norm(pred, 2).pow(2) / batch_size;
            }
            reg_loss /= predictions.size(); // Average over outputs
            total_loss += reg_weight * reg_loss;
        }

        return total_loss;
    }

    auto MultiLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        vector<torch::Tensor> predictions;
        vector<torch::Tensor> targets;
        vector<float> weights;
        vector<bool> is_classification;
        return xt::losses::multi_loss(predictions, targets, weights, is_classification);
    }
}
