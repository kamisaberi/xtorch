#include "include/losses/elastic_face.h"

namespace xt::losses
{
    torch::Tensor elastic_face(const torch::Tensor& logits, const torch::Tensor& labels, float s, float m_mean,
                               float m_std, float eps = 1e-6)
    {
        // Ensure inputs are on the same device
        auto device = logits.device();
        auto labels_device = labels.to(device);

        // Input validation
        TORCH_CHECK(logits.dim() == 2, "Logits must be 2D (batch_size, num_classes)");
        TORCH_CHECK(labels.dim() == 1, "Labels must be 1D (batch_size)");
        TORCH_CHECK(logits.size(0) == labels.size(0), "Batch sizes must match");
        TORCH_CHECK(m_mean >= 0, "Mean margin must be non-negative");
        TORCH_CHECK(m_std >= 0, "Margin standard deviation must be non-negative");

        // Generate random margins from normal distribution
        torch::Tensor margins = torch::normal(m_mean, m_std, {logits.size(0)});
        margins = torch::clamp(margins, 0.0, m_mean + 2 * m_std);

        // Create one-hot encoding of labels
        auto one_hot = torch::zeros_like(logits).scatter_(1, labels_device.view({-1, 1}), 1.0);

        // Compute cos(theta + m) for target classes
        torch::Tensor cos_theta = logits;
        torch::Tensor sin_theta = torch::sqrt(1.0 - torch::pow(cos_theta, 2) + eps);
        torch::Tensor cos_theta_m = cos_theta * torch::cos(margins.view({-1, 1})) -
            sin_theta * torch::sin(margins.view({-1, 1}));

        // Apply margin only to target classes
        torch::Tensor modified_logits = s * (one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta);

        // Compute cross-entropy loss
        torch::Tensor loss = torch::nn::functional::cross_entropy(modified_logits, labels_device);
        return loss;
    }

    auto ElasticFace::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::elastic_face(torch::zeros(10), torch::zeros(10), 0.0f, 0.0f, 0.0f);
    }
}
