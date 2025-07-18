#include <transforms/image/random_auto_contrast.h>


// --- Example Main (for testing) ---
// #include "transforms/image/random_auto_contrast.h"
// #include <iostream>
// #include <opencv2/highgui.hpp>
// #include "utils/image_conversion.h"
//
// int main() {
//     // 1. Create a low-contrast image.
//     // A gradient with a very small range to demonstrate auto-contrast.
//     torch::Tensor low_contrast_tensor = torch::linspace(0.4, 0.6, 256).view({1, -1}).repeat({3, 256, 1});
//
//     // 2. Apply RandomAutoContrast with different cutoffs and probabilities.
//     std::cout << "--- Applying RandomAutoContrast ---" << std::endl;
//
//     // Example 1: Strong AutoContrast (0 cutoff means full stretch, p=1.0 means always apply)
//     xt::transforms::image::RandomAutoContrast autocontrast_strong(0.0, 1.0);
//     torch::Tensor contrasted_strong = std::any_cast<torch::Tensor>(autocontrast_strong.forward({low_contrast_tensor}));
//     cv::Mat contrasted_strong_mat = xt::utils::image::tensor_to_mat_8u(contrasted_strong);
//     cv::imwrite("autocontrast_strong.png", contrasted_strong_mat);
//     std::cout << "Saved autocontrast_strong.png" << std::endl;
//
//     // Example 2: Moderate AutoContrast (0.1 cutoff, p=1.0)
//     xt::transforms::image::RandomAutoContrast autocontrast_moderate(0.1, 1.0);
//     torch::Tensor contrasted_moderate = std::any_cast<torch::Tensor>(autocontrast_moderate.forward({low_contrast_tensor}));
//     cv::Mat contrasted_moderate_mat = xt::utils::image::tensor_to_mat_8u(contrasted_moderate);
//     cv::imwrite("autocontrast_moderate.png", contrasted_moderate_mat);
//     std::cout << "Saved autocontrast_moderate.png" << std::endl;
//
//     return 0;
// }


namespace xt::transforms::image {

    RandomAutoContrast::RandomAutoContrast() : RandomAutoContrast(0.0, 0.5) {}

    RandomAutoContrast::RandomAutoContrast(double cutoff, double p)
        : cutoff_(cutoff), p_(p) {

        if (cutoff_ < 0.0 || cutoff_ > 0.5) {
            throw std::invalid_argument("Cutoff must be between 0.0 and 0.5.");
        }
        if (p_ < 0.0 || p_ > 1.0) {
            throw std::invalid_argument("Probability p must be between 0.0 and 1.0.");
        }

        std::random_device rd;
        gen_.seed(rd());
    }

    auto RandomAutoContrast::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- Probability Check ---
        std::uniform_real_distribution<> prob_dist(0.0, 1.0);
        if (prob_dist(gen_) >= p_) {
            return tensors.begin()[0]; // Return the original tensor
        }

        // --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("RandomAutoContrast::forward received an empty list.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to RandomAutoContrast is not defined.");
        }

        // --- Convert to OpenCV Mat (8-bit) ---
        cv::Mat input_mat_8u = xt::utils::image::tensor_to_mat_8u(input_tensor);

        // --- Apply AutoContrast Per Channel ---
        // This logic works for both single-channel (grayscale) and multi-channel (color) images.
        std::vector<cv::Mat> channels_out;
        for (int i = 0; i < input_mat_8u.channels(); ++i) {
            // Correctly extract the i-th channel
            cv::Mat single_channel;
            cv::extractChannel(input_mat_8u, single_channel, i);

            // --- Manual Percentile Calculation and Stretching ---
            // 1. Find min and max intensity values based on the histogram
            double min_val_d, max_val_d;
            cv::minMaxLoc(single_channel, &min_val_d, &max_val_d);

            // If min and max are already full range, or cutoff is high, we might not need to stretch.
            // But the percentile method is more robust.
            uchar min_val_p = static_cast<uchar>(min_val_d);
            uchar max_val_p = static_cast<uchar>(max_val_d);

            // 2. Compute histogram if cutoff is used
            if (cutoff_ > 0.0) {
                cv::Mat hist;
                int histSize = 256;
                float range[] = {0, 256};
                const float* histRange = {range};
                cv::calcHist(&single_channel, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);

                long total_pixels = single_channel.total();
                long cumulative_sum = 0;
                long lower_bound = static_cast<long>(total_pixels * cutoff_);
                long upper_bound = static_cast<long>(total_pixels * (1.0 - cutoff_));

                // Find the new lower bound (min_val_p)
                for (int k = 0; k < histSize; ++k) {
                    cumulative_sum += static_cast<long>(hist.at<float>(k));
                    if (cumulative_sum >= lower_bound) {
                        min_val_p = k;
                        break;
                    }
                }

                // Find the new upper bound (max_val_p)
                cumulative_sum = 0;
                 for (int k = 0; k < histSize; ++k) {
                    cumulative_sum += static_cast<long>(hist.at<float>(k));
                    if (cumulative_sum >= upper_bound) {
                        max_val_p = k;
                        break;
                    }
                }
            }

            // 3. Apply linear transformation (stretch)
            cv::Mat stretched_channel;
            if (max_val_p <= min_val_p) {
                // If the range is zero or inverted, just use the original channel
                stretched_channel = single_channel;
            } else {
                // Formula: dst = alpha * src + beta
                // We want: output = (input - min_val) * (255.0 / (max_val - min_val))
                double alpha = 255.0 / (max_val_p - min_val_p);
                double beta = -min_val_p * alpha;
                single_channel.convertTo(stretched_channel, CV_8U, alpha, beta);
            }
            channels_out.push_back(stretched_channel);
        }

        // Merge the processed channels back into a single image
        cv::Mat output_mat_8u;
        if (channels_out.size() > 1) {
            cv::merge(channels_out, output_mat_8u);
        } else {
            output_mat_8u = channels_out[0];
        }

        // --- Convert back to LibTorch Tensor (Float) ---
        torch::Tensor output_tensor = xt::utils::image::mat_to_tensor_float(output_mat_8u);

        return output_tensor;
    }

} // namespace xt::transforms::image