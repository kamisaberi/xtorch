#include "include/transforms/image/glass_blur.h"



// #include "transforms/image/glass_blur.h"
// #include <iostream>
//
// int main() {
//     // 1. Create or load a dummy image tensor
//     torch::Tensor image = torch::rand({3, 224, 224});
//
//     std::cout << "Original image shape: " << image.sizes() << std::endl;
//
//     // 2. Instantiate the transform with parameters for a noticeable effect
//     xt::transforms::image::GlassBlur transformer(
//         /*sigma=*/1.5,
//         /*max_delta=*/2,
//         /*iterations=*/2
//     );
//
//     // 3. Apply the transform
//     std::any result_any = transformer.forward({image});
//     torch::Tensor blurred_image = std::any_cast<torch::Tensor>(result_any);
//
//     // 4. Check the output
//     std::cout << "Glass-blurred image shape: " << blurred_image.sizes() << std::endl;
//
//     // You could now save the original and blurred images to visually inspect the effect.
//     // The output will look like the original image viewed through a piece of textured glass.
//     // cv::Mat original_mat = xt::utils::image::tensor_to_mat_8u(image);
//     // cv::imwrite("original_for_glassblur.png", original_mat);
//     //
//     // cv::Mat blurred_mat = xt::utils::image::tensor_to_mat_8u(blurred_image);
//     // cv::imwrite("glass_blurred_image.png", blurred_mat);
//
//     return 0;
// }


namespace xt::transforms::image {

    GlassBlur::GlassBlur() : sigma_(0.7), max_delta_(1), iterations_(2) {}

    GlassBlur::GlassBlur(double sigma, int max_delta, int iterations)
        : sigma_(sigma), max_delta_(max_delta), iterations_(iterations) {

        if (sigma_ <= 0 || max_delta_ <= 0 || iterations_ <= 0) {
            throw std::invalid_argument("GlassBlur parameters must be positive.");
        }
    }

    auto GlassBlur::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation and Conversion to Mat ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("GlassBlur::forward received an empty list of tensors.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to GlassBlur is not defined.");
        }

        // We work with float Mats [0, 1] to preserve precision.
        cv::Mat input_mat = xt::utils::image::tensor_to_mat_local(input_tensor);

        // 2. --- Perform Pixel Shuffling ---
        cv::Mat shuffled_mat = input_mat.clone();
        int height = input_mat.rows;
        int width = input_mat.cols;

        for (int i = 0; i < iterations_; ++i) {
            cv::Mat temp_mat = shuffled_mat.clone();

            // For each pixel, choose a random nearby pixel to copy from
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    // Generate random offset
                    int dx = cv::theRNG().uniform(-max_delta_, max_delta_ + 1);
                    int dy = cv::theRNG().uniform(-max_delta_, max_delta_ + 1);

                    // Clamp coordinates to stay within image bounds
                    int new_x = std::min(width - 1, std::max(0, x + dx));
                    int new_y = std::min(height - 1, std::max(0, y + dy));

                    // Copy the pixel from the source (temp_mat) to the destination (shuffled_mat)
                    if (input_mat.channels() == 3) {
                        shuffled_mat.at<cv::Vec3f>(y, x) = temp_mat.at<cv::Vec3f>(new_y, new_x);
                    } else { // Grayscale
                        shuffled_mat.at<float>(y, x) = temp_mat.at<float>(new_y, new_x);
                    }
                }
            }
        }

        // 3. --- Apply Gaussian Blur to the Shuffled Image ---
        cv::Mat blurred_mat;
        // The kernel size is determined from sigma by OpenCV if set to (0,0)
        cv::GaussianBlur(shuffled_mat, blurred_mat, cv::Size(0, 0), sigma_);

        // 4. --- Clamp and Convert back to LibTorch Tensor ---
        // Clamp values just in case, though they should be fine.
        cv::patchNaNs(blurred_mat, 0); // Handle potential NaN from blurring
        cv::max(blurred_mat, 0, blurred_mat);
        cv::min(blurred_mat, 1, blurred_mat);

        torch::Tensor output_tensor = xt::utils::image::mat_to_tensor_local(blurred_mat);

        return output_tensor;
    }

} // namespace xt::transforms::image