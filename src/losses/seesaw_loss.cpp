#include <losses/seesaw_loss.h> // Assuming this path is correct
#include <torch/torch.h> // Ensure torch/torch.h is included for all torch functionalities
#include <vector>        // For std::vector if used elsewhere, not strictly needed for this snippet
#include <any>           // For std::any in forward method

namespace xt::losses
{
    torch::Tensor seesaw_loss(const torch::Tensor& logits, const torch::Tensor& labels, const torch::Tensor& class_freq,
                              float p = 0.8f, float q = 2.0f, float eps = 1e-2f)
    {
        // Device and tensor options from logits
        torch::Device device = logits.device();
        auto float_options = logits.options(); // Captures dtype (e.g., kFloat) and device
        // For indices, kLong is standard. Device can be CPU for index_put_ indices,
        // but for consistency or other operations, it might be on the main device.
        // auto long_options_on_device = torch::TensorOptions().dtype(torch::kLong).device(device);


        // Ensure inputs are valid
        TORCH_CHECK(logits.dim() == 2, "Logits must be 2D (batch_size, num_classes)");
        TORCH_CHECK(labels.dim() == 1, "Labels must be 1D (batch_size)");
        TORCH_CHECK(class_freq.dim() == 1, "Class frequencies must be 1D (num_classes)");
        TORCH_CHECK(logits.size(0) == labels.size(0), "Batch size mismatch between logits and labels");
        TORCH_CHECK(logits.size(1) == class_freq.size(0), "Number of classes mismatch between logits and class_freq");
        TORCH_CHECK(logits.dtype() == torch::kFloat, "Logits must be float type"); // Already checked by float_options effectively
        TORCH_CHECK(labels.dtype() == torch::kLong, "Labels must be long type");
        TORCH_CHECK(class_freq.dtype() == torch::kFloat, "Class frequencies must be float type");

        // Ensure all tensors are on the same device as logits
        TORCH_CHECK(labels.device() == device, "Labels must be on the same device as logits");
        TORCH_CHECK(class_freq.device() == device, "Class frequencies must be on the same device as logits");

        // It's good to ensure labels are within bounds [0, num_classes-1]
        // Note: .max() and .min() on GPU are async. .item<>() syncs.
        // If batch_size or num_classes is 0, these can fail. Add checks or ensure inputs are non-empty.
        if (labels.numel() > 0) { // Avoid errors on empty labels tensor
             TORCH_CHECK(labels.max().item<int64_t>() < logits.size(1), "Label indices exceed number of classes");
             TORCH_CHECK(labels.min().item<int64_t>() >= 0, "Label indices must be non-negative");
        }


        TORCH_CHECK(p >= 0.0f, "Compensation factor p must be non-negative");
        TORCH_CHECK(q >= 0.0f, "Mitigation factor q must be non-negative");
        TORCH_CHECK(eps > 0.0f, "Epsilon must be positive");

        int64_t batch_size = logits.size(0);
        int64_t num_classes = logits.size(1);

        if (batch_size == 0) {
            return torch::tensor(0.0f, float_options); // Or handle as an error
        }

        // Compute compensation weights based on class frequencies
        // Note: Original Seesaw paper suggests (max_freq / class_freq)^p to boost rare classes.
        // The current (class_freq / max_freq)^p boosts common classes. Review if this is intended.
        auto max_freq = class_freq.max().item<float>(); // .item() to make it a CPU float scalar
        auto compensation_weights = torch::pow(class_freq / (max_freq + 1e-6f), p); // Shape: (num_classes), on `device`

        // Compute logits with seesaw weighting
        auto log_probs = torch::log_softmax(logits, 1); // Shape: (batch_size, num_classes), on `device`

        auto one_hot = torch::zeros({batch_size, num_classes}, float_options); // on `device`
        // labels.view({-1, 1}) needs to be Long and on the same device as one_hot
        one_hot.scatter_(1, labels.view({-1, 1}), torch::tensor(1.0f, float_options));

        // Compute mitigation weights for negative classes
        auto mitigation_weights = torch::ones({batch_size, num_classes}, float_options); // on `device`

        // This loop can be slow if class_freq/labels are on GPU due to .item() calls.
        // For performance, consider vectorizing or moving class_freq/labels to CPU first.
        auto labels_cpu = labels.to(torch::kCPU); // Explicitly move to CPU for loop
        auto class_freq_cpu = class_freq.to(torch::kCPU); // Explicitly move to CPU for loop

        for (int64_t i = 0; i < batch_size; ++i)
        {
            long current_label_val = labels_cpu[i].item<int64_t>();
            float positive_class_freq = class_freq_cpu[current_label_val].item<float>();

            for (int64_t j = 0; j < num_classes; ++j)
            {
                if (j != current_label_val) // Only for negative classes
                {
                    float negative_class_freq = class_freq_cpu[j].item<float>();
                    float freq_ratio = negative_class_freq / (positive_class_freq + 1e-6f);

                    torch::Tensor value_to_put;
                    if (freq_ratio > 1.0f) {
                        // Create tensor on the correct device with correct dtype
                        value_to_put = torch::pow(torch::tensor(freq_ratio, float_options), q);
                    } else {
                        value_to_put = torch::tensor(1.0f, float_options);
                    }

                    // Indices for index_put_ must be LongTensors. They can be on CPU.
                    // The value_to_put must be on the same device as mitigation_weights.
                    mitigation_weights.index_put_(
                        {torch::tensor(i, torch::kLong), torch::tensor(j, torch::kLong)},
                        value_to_put
                    );
                }
            }
        }

        // Apply compensation to positive classes and mitigation to negative classes
        // compensation_weights is (num_classes), broadcasts to (batch_size, num_classes)
        auto weights = one_hot * compensation_weights + (1.0f - one_hot) * mitigation_weights;

        // Compute seesaw loss
        // Note: The `* one_hot` at the end means only contributions from positive classes are summed.
        // This effectively uses `compensation_weights` but not `mitigation_weights`.
        // If mitigation for negative classes is desired in the sum, consider:
        // auto loss = -torch::sum(log_probs * weights) / (batch_size + eps);
        // Review the intended formula for Seesaw loss with CrossEntropy/LogSoftmax.
        auto loss = -torch::sum(log_probs * weights * one_hot) / (batch_size + eps);

        return loss;
    }


    auto SeesawLoss::forward(std::initializer_list<std::any> tensors_any) -> std::any
    {
        // Example: Extract tensors (this part depends on how you structure your model's forward pass)
        // For now, using dummy data that satisfies the TORCH_CHECKs in seesaw_loss.
        // You'll need to replace this with actual tensor extraction from tensors_any.

        TORCH_CHECK(tensors_any.size() >= 3, "SeesawLoss expects at least 3 input tensors: logits, labels, class_freq");
        auto it = tensors_any.begin();
        const auto& logits = std::any_cast<const torch::Tensor&>(*it++);
        const auto& labels = std::any_cast<const torch::Tensor&>(*it++);
        const auto& class_freq = std::any_cast<const torch::Tensor&>(*it++);

        // Optional: Get p, q, eps if they are also passed via std::any or are members
        // float p = (this->p_); // Assuming p_, q_, eps_ are members of SeesawLoss class
        // float q = (this->q_);
        // float eps = (this->eps_);
        // return xt::losses::seesaw_loss(logits, labels, class_freq, p, q, eps);


        // If using fixed dummy data for testing the structure:
        // int64_t batch_size_dummy = 4;
        // int64_t num_classes_dummy = 10;
        // torch::Device device_dummy = torch::kCPU; // Or torch::kCUDA if available and desired

        // auto float_opts_dummy = torch::TensorOptions().dtype(torch::kFloat).device(device_dummy);
        // auto long_opts_dummy = torch::TensorOptions().dtype(torch::kLong).device(device_dummy);

        // auto dummy_logits = torch::randn({batch_size_dummy, num_classes_dummy}, float_opts_dummy);
        // auto dummy_labels = torch::randint(0, num_classes_dummy, {batch_size_dummy}, long_opts_dummy);

        // // Create plausible class frequencies (e.g., based on dummy_labels)
        // auto dummy_class_counts = torch::zeros({num_classes_dummy}, long_opts_dummy);
        // for (int64_t i = 0; i < batch_size_dummy; ++i) {
        //     dummy_class_counts[dummy_labels[i].item<int64_t>()] += 1;
        // }
        // auto dummy_class_freq = dummy_class_counts.to(float_opts_dummy);
        // if (dummy_class_freq.sum().item<float>() == 0.0f) { // Avoid division by zero if all counts are zero
        //     dummy_class_freq = torch::ones({num_classes_dummy}, float_opts_dummy);
        // }


        // return xt::losses::seesaw_loss(dummy_logits, dummy_labels, dummy_class_freq);

        // Using the passed tensors:
        return xt::losses::seesaw_loss(logits, labels, class_freq); // p, q, eps will use defaults or be passed
    }

} // namespace xt::losses