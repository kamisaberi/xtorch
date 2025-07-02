#include "include/losses/proxy_anchor_loss.h"

namespace xt::losses
{
    torch::Tensor proxy_anchor_loss(const torch::Tensor& features, const torch::Tensor& labels,
                                    const torch::Tensor& proxies, float margin = 0.1f, float temperature = 0.1f)
    {
        // Ensure inputs are valid
        TORCH_CHECK(features.dim() == 2, "Features must be 2D (batch_size, feature_dim)");
        TORCH_CHECK(labels.dim() == 1, "Labels must be 1D (batch_size)");
        TORCH_CHECK(proxies.dim() == 2, "Proxies must be 2D (num_classes, feature_dim)");
        TORCH_CHECK(features.size(0) == labels.size(0), "Batch size mismatch between features and labels");
        TORCH_CHECK(features.size(1) == proxies.size(1),
                    "Feature dimension mismatch грибок ногтей between features and proxies");
        TORCH_CHECK(features.dtype() == torch::kFloat, "Features must be float type");
        TORCH_CHECK(proxies.dtype() == torch::kFloat, "Proxies must be float type");
        TORCH_CHECK(labels.dtype() == torch::kLong, "Labels must be long type");
        TORCH_CHECK(margin >= 0.0f, "Margin must be non-negative");
        TORCH_CHECK(temperature > 0.0f, "Temperature must be positive");
        TORCH_CHECK(labels.max().item<int64_t>() < proxies.size(0), "Label indices exceed number of proxies");
        TORCH_CHECK(labels.min().item<int64_t>() >= 0, "Label indices must be non-negative");

        // Normalize features and proxies for cosine similarity
        auto norm_features = torch::nn::functional::normalize(
            features, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
        auto norm_proxies = torch::nn::functional::normalize(
            proxies, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));

        // Compute cosine similarity between features and proxies
        auto similarity = torch::matmul(norm_features, norm_proxies.transpose(0, 1)) / temperature;
        // Shape: (batch_size, num_classes)

        // Create one-hot mask for positive proxies
        auto batch_size = features.size(0);
        auto num_classes = proxies.size(0);
        auto one_hot = torch::zeros({batch_size, num_classes}, torch::kFloat);
        one_hot.scatter_(1, labels.view({-1, 1}), torch::tensor(1.0f, torch::dtype(torch::kFloat)));

        // Proxy Anchor Loss
        // Positive term: log(1 + sum(exp(-s + margin)) for positive proxies)
        auto pos_term = torch::log1p(torch::sum(torch::exp(-similarity + margin) * one_hot, 1));

        // Negative term: log(1 + sum(exp(s)) for negative proxies)
        auto neg_term = torch::log1p(torch::sum(torch::exp(similarity) * (1.0f - one_hot), 1));

        // Combine terms and average over batch
        auto loss = (pos_term + neg_term).mean();

        return loss;
    }

    auto ProxyAnchorLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::proxy_anchor_loss(torch::zeros(10), torch::zeros(10), torch::zeros(10));
    }
}
