#include <losses/metric_mixup_loss.h>

namespace xt::losses
{
    /**
 * Metric Mixup Loss function for metric learning with Mixup augmentation.
 * Combines contrastive metric learning with Mixup interpolation for improved generalization.
 * @param embeddings Input embeddings, shape: [batch_size, embedding_dim]
 * @param labels Ground truth labels, shape: [batch_size]
 * @param margin Margin for negative pairs, default: 1.0
 * @param alpha Alpha parameter for Beta distribution in Mixup, default: 1.0
 * @param class_weights Weights for each class to handle imbalance, shape: [num_classes], default: nullptr (uniform)
 * @param eps Small value for numerical stability, default: 1e-6
 * @return Scalar tensor containing the Metric Mixup Loss
 */
    torch::Tensor metric_mixup_loss(const torch::Tensor& embeddings, const torch::Tensor& labels, float margin = 1.0,
                                    float alpha = 1.0,
                                    const torch::Tensor& class_weights = torch::Tensor(),
                                    float eps = 1e-6)
    {
        // Ensure inputs are on the same device
        auto device = embeddings.device();
        auto labels_device = labels.to(device).to(torch::kLong); // Ensure labels are int64

        // Input validation
        TORCH_CHECK(embeddings.dim() == 2, "Embeddings must be 2D (batch_size, embedding_dim)");
        TORCH_CHECK(labels.dim() == 1, "Labels must be 1D (batch_size)");
        TORCH_CHECK(embeddings.size(0) == labels.size(0), "Batch sizes must match");
        TORCH_CHECK(margin >= 0, "Margin must be non-negative");
        TORCH_CHECK(alpha > 0, "Alpha must be positive");
        if (class_weights.defined())
        {
            TORCH_CHECK(class_weights.dim() == 1, "Class weights must be 1D (num_classes)");
            TORCH_CHECK((class_weights >= 0).all().item<bool>(), "Class weights must be non-negative");
        }

        int batch_size = embeddings.size(0);

        // Compute number of classes safely
        TORCH_CHECK((labels_device >= 0).all().item<bool>(), "Labels must be non-negative");
        int64_t num_classes = labels_device.max().item<int64_t>() + 1;

        // Generate Mixup interpolation factors from Beta distribution
        std::random_device rd;
        std::mt19937 gen(rd());
        std::gamma_distribution<float> gamma_dist(alpha, 1.0);
        std::vector<float> lambdas(batch_size);
        for (int i = 0; i < batch_size; ++i)
        {
            float gamma1 = gamma_dist(gen);
            float gamma2 = gamma_dist(gen);
            lambdas[i] = gamma1 / (gamma1 + gamma2); // Beta distribution
        }
        torch::Tensor lambda_tensor = torch::tensor(lambdas, torch::TensorOptions().device(device)).unsqueeze(1);

        // Randomly shuffle indices for Mixup pairs
        torch::Tensor indices = torch::randperm(batch_size, torch::TensorOptions().dtype(torch::kLong).device(device));

        // Create mixed embeddings: lambda * emb[i] + (1-lambda) * emb[indices[i]]
        torch::Tensor mixed_embeddings = lambda_tensor * embeddings + (1.0 - lambda_tensor) * embeddings.
            index({indices});

        // Create one-hot labels
        torch::Tensor one_hot_labels = torch::zeros({batch_size, num_classes},
                                                    torch::TensorOptions().dtype(torch::kFloat).device(device));
        one_hot_labels.scatter_(1, labels_device.unsqueeze(1),
                                torch::tensor(1.0, torch::TensorOptions().device(device)));

        // Create soft labels for Mixup: lambda * label[i] + (1-lambda) * label[indices[i]]
        torch::Tensor mixed_labels = lambda_tensor * one_hot_labels + (1.0 - lambda_tensor) * one_hot_labels.index({
            indices
        });

        // Compute pairwise Euclidean distances for mixed embeddings
        torch::Tensor emb_norm = mixed_embeddings.norm(2, 1, true); // Shape: [batch_size, 1]
        torch::Tensor dot_product = torch::matmul(mixed_embeddings, mixed_embeddings.transpose(0, 1));
        // Shape: [batch_size, batch_size]
        torch::Tensor squared_norm = emb_norm * emb_norm.transpose(0, 1); // Shape: [batch_size, batch_size]
        torch::Tensor distances = torch::sqrt(
            (squared_norm - 2 * dot_product + squared_norm.transpose(0, 1)).clamp_min(eps));
        // Shape: [batch_size, batch_size]

        // Compute similarity matrix for soft labels (cosine similarity between mixed labels)
        torch::Tensor label_similarities = torch::matmul(mixed_labels, mixed_labels.transpose(0, 1));
        // Shape: [batch_size, batch_size]

        // Mask to exclude self-comparisons (diagonal)
        torch::Tensor mask = torch::ones_like(label_similarities) - torch::eye(
            batch_size, torch::TensorOptions().device(device));

        // Compute positive pair loss (minimize distance for similar labels)
        torch::Tensor pos_mask = (label_similarities > 0.5).to(torch::kFloat); // Threshold for positive pairs
        torch::Tensor pos_loss = pos_mask * distances * mask;
        torch::Tensor pos_count = pos_mask.sum().clamp_min(eps);
        pos_loss = pos_loss.sum() / pos_count;

        // Compute negative pair loss (max(0, margin - distance) for dissimilar labels)
        torch::Tensor neg_mask = (label_similarities <= 0.5).to(torch::kFloat);
        torch::Tensor neg_loss = neg_mask * torch::relu(margin - distances) * mask;
        torch::Tensor neg_count = neg_mask.sum().clamp_min(eps);
        neg_loss = neg_loss.sum() / neg_count;

        // Apply class weights based on original labels
        torch::Tensor pair_weights = torch::ones_like(label_similarities);
        if (class_weights.defined())
        {
            torch::Tensor sample_weights = class_weights.index({labels_device}).to(device);
            pair_weights = sample_weights.unsqueeze(1) * sample_weights.unsqueeze(0);
        }

        // Combine positive and negative losses with class weights and label similarities
        torch::Tensor loss = (pos_loss + neg_loss) * (pair_weights * label_similarities).mean();

        return loss;
    }

    auto MetricMixupLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::metric_mixup_loss(torch::zeros(10), torch::zeros(10));
    }
}
