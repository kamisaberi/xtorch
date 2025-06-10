#include "include/losses/upit.h"

namespace xt::losses
{
    torch::Tensor upit_loss(const torch::Tensor& pred_signals, const torch::Tensor& gt_signals)
    {
        // Ensure inputs are valid
        TORCH_CHECK(pred_signals.dim() == 3, "Predicted signals must be 3D (batch_size, num_speakers, signal_length)");
        TORCH_CHECK(gt_signals.dim() == 3, "Ground truth signals must be 3D (batch_size, num_speakers, signal_length)");
        TORCH_CHECK(pred_signals.size(0) == gt_signals.size(0), "Batch size mismatch");
        TORCH_CHECK(pred_signals.size(1) == gt_signals.size(1), "Number of speakers mismatch");
        TORCH_CHECK(pred_signals.size(2) == gt_signals.size(2), "Signal length mismatch");
        TORCH_CHECK(pred_signals.dtype() == torch::kFloat, "Predicted signals must be float type");
        TORCH_CHECK(gt_signals.dtype() == torch::kFloat, "Ground truth signals must be float type");

        int64_t batch_size = pred_signals.size(0);
        int64_t num_speakers = pred_signals.size(1);
        torch::Tensor loss = torch::zeros({batch_size}, torch::kFloat);

        // Generate all permutations for num_speakers (example for 2 or 3 speakers)
        std::vector<std::vector<int64_t>> permutations;
        if (num_speakers == 2)
        {
            permutations = {{0, 1}, {1, 0}};
        }
        else if (num_speakers == 3)
        {
            permutations = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
        }
        else
        {
            TORCH_CHECK(false, "UPIT loss supports only 2 or 3 speakers for simplicity");
        }

        for (int64_t b = 0; b < batch_size; ++b)
        {
            auto pred = pred_signals[b]; // Shape: (num_speakers, signal_length)
            auto gt = gt_signals[b]; // Shape: (num_speakers, signal_length)
            torch::Tensor min_loss = torch::tensor(std::numeric_limits<float>::max(), torch::kFloat);

            // Try each permutation
            for (const auto& perm : permutations)
            {
                torch::Tensor permuted_pred = torch::empty_like(pred);
                for (size_t i = 0; i < perm.size(); ++i)
                {
                    permuted_pred[i] = pred[perm[i]];
                }

                // Compute MSE for this permutation
                auto mse = torch::mean(torch::pow(permuted_pred - gt, 2));
                min_loss = torch::min(min_loss, mse);
            }

            loss.index_put_({b}, min_loss);
        }

        // Return mean loss over batch
        return loss.mean();
    }

    auto UPITLoss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::losses::upit_loss(torch::zeros(10),torch::zeros(10));
    }
}
