#include <transforms/signal/sliding_window_cmn.h>


/*
// Example Usage (goes in a main.cpp or test file)
#include <iostream>

int main() {
    // 1. Create a dummy feature tensor (e.g., MFCCs).
    // Let's make it 40 features over 1000 time steps.
    // Add a DC offset to simulate channel effects.
    torch::Tensor features = torch::randn({40, 1000}) + 5.0;
    std::cout << "Mean of first feature before CMN: " << features[0].mean().item<float>() << std::endl;

    // 2. Create the SlidingWindowCMN transform.
    // Use a window of 301 frames, centered, and normalize both mean and variance.
    xt::transforms::signal::SlidingWindowCMN cmn_transform(301, 100, true, true, true);

    // 3. Apply the transform.
    torch::Tensor normalized_features = std::any_cast<torch::Tensor>(
        cmn_transform.forward({features})
    );

    // 4. Verify the output.
    // The mean of any local window in the output should be close to 0.
    std::cout << "Mean of first feature after CMN (window 300-600): "
              << normalized_features.slice(1, 300, 600)[0].mean().item<float>()
              << std::endl;
    std::cout << "Std of first feature after CMN (window 300-600): "
              << normalized_features.slice(1, 300, 600)[0].std().item<float>()
              << std::endl;
    // The mean should be near 0 and std near 1.

    return 0;
}
*/

namespace xt::transforms::signal {

    SlidingWindowCMN::SlidingWindowCMN(
            int cmn_window,
            int min_cmn_window,
            bool normalize_mean,
            bool normalize_variance,
            bool center,
            double p)
            : cmn_window_(cmn_window),
              min_cmn_window_(min_cmn_window),
              normalize_mean_(normalize_mean),
              normalize_variance_(normalize_variance),
              center_(center),
              p_(p) {

        if (cmn_window_ < 1) {
            throw std::invalid_argument("cmn_window must be at least 1.");
        }
        if (min_cmn_window_ > cmn_window_ || min_cmn_window_ < 1) {
            throw std::invalid_argument("min_cmn_window is invalid.");
        }

        std::random_device rd;
        random_engine_.seed(rd());
    }

    auto SlidingWindowCMN::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- 1. Input Validation and Probability ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("SlidingWindowCMN::forward received an empty list.");
        }
        torch::Tensor feats = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!feats.defined() || feats.dim() != 2) {
            throw std::invalid_argument("Input must be a 2D feature tensor (feats, time).");
        }

        std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
        if (prob_dist(random_engine_) > p_ || (!normalize_mean_ && !normalize_variance_)) {
            return feats; // Skip transform
        }

        // --- 2. Setup for Sliding Window ---
        // Transpose to (time, feats) for easier slicing. We'll transpose back at the end.
        feats = feats.transpose(0, 1).contiguous();
        long num_frames = feats.size(0);
        long num_feats = feats.size(1);
        auto device = feats.device();
        auto dtype = feats.dtype();
        auto eps = torch::tensor(1e-8, torch::kFloat32).to(device);

        auto normalized_feats = torch::empty_like(feats);

        // --- 3. Efficient Sliding Window Calculation ---
        int left_context = center_ ? (cmn_window_ - 1) / 2 : cmn_window_ - 1;
        int right_context = center_ ? cmn_window_ / 2 : 0;

        // Calculate the first window's statistics
        long last_pos = std::min((long)cmn_window_, num_frames);
        auto first_window = feats.slice(0, 0, last_pos);
        auto sum = first_window.sum(0);
        auto sum_sq = first_window.pow(2).sum(0);

        for (long t = 0; t < num_frames; ++t) {
            long window_start = std::max(0L, t - left_context);
            long window_end = std::min(num_frames, t + right_context + 1);
            long window_size = window_end - window_start;

            if (window_size < min_cmn_window_) {
                // Not enough context, use statistics from the first valid window
                window_size = std::min((long)cmn_window_, num_frames);
                // No change to `sum` or `sum_sq`
            } else {
                // Efficiently update the running sum by adding/subtracting frames at the edges
                if (t > 0) {
                    long prev_window_start = std::max(0L, t - 1 - left_context);
                    long prev_window_end = std::min(num_frames, t - 1 + right_context + 1);
                    if (window_start > prev_window_start) {
                        sum -= feats[prev_window_start];
                        sum_sq -= feats[prev_window_start].pow(2);
                    }
                    if (window_end > prev_window_end) {
                        sum += feats[window_end - 1];
                        sum_sq += feats[window_end - 1].pow(2);
                    }
                }
            }

            // --- 4. Normalize the current frame ---
            auto current_frame = feats[t];

            if (normalize_mean_) {
                auto mean = sum / window_size;
                current_frame = current_frame - mean;
            }

            if (normalize_variance_) {
                auto mean = sum / window_size;
                auto variance = (sum_sq / window_size) - mean.pow(2);
                auto std_dev = torch::sqrt(torch::clamp_min(variance, 0.0) + eps);
                current_frame = current_frame / std_dev;
            }

            normalized_feats[t] = current_frame;
        }

        // Transpose back to (feats, time) before returning.
        return normalized_feats.transpose(0, 1).contiguous();
    }

} // namespace xt::transforms::signal