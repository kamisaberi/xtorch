#include "include/losses/seesaw_loss.h"

namespace xt::losses
{
    torch::Tensor seesaw_loss(const torch::Tensor& logits, const torch::Tensor& labels, const torch::Tensor& class_freq,
                              float p = 0.8f, float q = 2.0f, float eps = 1e-2f)
    {
        // Ensure inputs are valid
        TORCH_CHECK(logits.dim() == 2, "Logits must be 2D (batch_size, num_classes)");
        TORCH_CHECK(labels.dim() == 1, "Labels must be 1D (batch_size)");
        TORCH_CHECK(class_freq.dim() == 1, "Class frequencies must be 1D (num_classes)");
        TORCH_CHECK(logits.size(0) == labels.size(0), "Batch size mismatch between logits and labels");
        TORCH_CHECK(logits.size(1) == class_freq.size(0), "Number of classes mismatch between logits and class_freq");
        TORCH_CHECK(logits.dtype() == torch::kFloat, "Logits must be float type");
        TORCH_CHECK(labels.dtype() == torch::kLong, "Labels must be long type");
        TORCH_CHECK(class_freq.dtype() == torch::kFloat, "Class frequencies must be float type");
        TORCH_CHECK(labels.max().item<int64_t>() < logits.size(1), "Label indices exceed number of classes");
        TORCH_CHECK(labels.min().item<int64_t>() >= 0, "Label indices must be non-negative");
        TORCH_CHECK(p >= 0.0f, "Compensation factor p must be non-negative");
        TORCH_CHECK(q >= 0.0f, "Mitigation factor q must be non-negative");
        TORCH_CHECK(eps > 0.0f, "Epsilon must be positive");

        int64_t batch_size = logits.size(0);
        int64_t num_classes = logits.size(1);

        // Compute compensation weights based on class frequencies
        auto max_freq = class_freq.max();
        auto compensation_weights = torch::pow(class_freq / (max_freq + 1e-6f), p); // Shape: (num_classes)

        // Compute logits with seesaw weighting
        auto log_probs = torch::log_softmax(logits, 1); // Shape: (batch_size, num_classes)
        auto one_hot = torch::zeros({batch_size, num_classes}, torch::kFloat);
        one_hot.scatter_(1, labels.view({-1, 1}), torch::tensor(1.0f, torch::dtype(torch::kFloat)));

        // Compute mitigation weights for negative classes
        auto mitigation_weights = torch::ones({batch_size, num_classes}, torch::kFloat);
        for (int64_t i = 0; i < batch_size; ++i)
        {
            for (int64_t j = 0; j < num_classes; ++j)
            {
                if (j != labels[i].item<int64_t>())
                {
                    float freq_ratio = class_freq[j].item<float>() / (class_freq[labels[i].item<int64_t>()].item<
                        float>() + 1e-6f);
                    mitigation_weights.index_put_(
                        {i, j}, freq_ratio > 1.0f ? torch::pow(torch::tensor(freq_ratio), q) : 1.0f);
                }
            }
        }

        // Apply compensation to positive classes and mitigation to negative classes
        auto weights = one_hot * compensation_weights + (1.0f - one_hot) * mitigation_weights;

        // Compute seesaw loss
        auto loss = -torch::sum(log_probs * weights * one_hot) / (batch_size + eps);

        return loss;
    }


    auto SeesawLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::seesaw_loss(torch::zeros(10));
    }
}
