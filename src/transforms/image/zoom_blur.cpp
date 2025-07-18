#include <transforms/image/zoom_blur.h>



// #include "transforms/image/zoom_blur.h"
// #include <iostream>
//
// int main() {
//     // 1. Create a dummy image tensor with distinct features
//     torch::Tensor image = torch::zeros({3, 224, 224});
//     // Add some "stars" to see the radial blur effect
//     image.index_put_({torch::indexing::Slice(), 50, 50}, 1.0);
//     image.index_put_({torch::indexing::Slice(), 100, 180}, 1.0);
//     image.index_put_({torch::indexing::Slice(), 180, 80}, 1.0);
//
//     // 2. Instantiate the transform with a noticeable zoom factor
//     xt::transforms::image::ZoomBlur blurer(
//         /*max_zoom=*/1.8f,
//         /*num_steps=*/10
//     );
//
//     // 3. Apply the transform
//     std::any result_any = blurer.forward({image});
//     torch::Tensor blurred_image = std::any_cast<torch::Tensor>(result_any);
//
//     // 4. Check the output
//     std::cout << "Original image shape: " << image.sizes() << std::endl;
//     std::cout << "Zoom-blurred image shape: " << blurred_image.sizes() << std::endl;
//
//     // You could save the original and blurred images to see the effect.
//     // The individual white "star" pixels will be blurred into streaks radiating
//     // from the center of the image.
//     // cv::Mat original_mat = xt::utils::image::tensor_to_mat_8u(image);
//     // cv::imwrite("original_for_zoom.png", original_mat);
//     //
//     // cv::Mat blurred_mat = xt::utils::image::tensor_to_mat_8u(blurred_image);
//     // cv::imwrite("zoom_blurred_image.png", blurred_mat);
//
//     return 0;
// }

namespace xt::transforms::image {

    ZoomBlur::ZoomBlur() : max_zoom_(1.5f), num_steps_(5) {}

    ZoomBlur::ZoomBlur(float max_zoom, int num_steps)
        : max_zoom_(max_zoom), num_steps_(num_steps) {

        if (max_zoom_ < 1.0f) {
            throw std::invalid_argument("ZoomBlur max_zoom must be >= 1.0.");
        }
        if (num_steps_ < 2) {
            throw std::invalid_argument("ZoomBlur num_steps must be at least 2 for a blur effect.");
        }
    }

    auto ZoomBlur::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation and Conversion to Mat ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("ZoomBlur::forward received an empty list of tensors.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to ZoomBlur is not defined.");
        }

        // We work with float Mats [0, 1] to preserve precision.
        cv::Mat input_mat = xt::utils::image::tensor_to_mat_local(input_tensor);
        cv::Mat blurred_mat = cv::Mat::zeros(input_mat.size(), input_mat.type());

        // 2. --- Create and Blend Scaled Images ---
        // Create a set of zoom factors from 1.0 up to max_zoom_
        std::vector<float> zoom_factors;
        for (int i = 0; i < num_steps_; ++i) {
            zoom_factors.push_back(1.0f + (max_zoom_ - 1.0f) * i / (num_steps_ - 1));
        }

        // The weight for each step in the blend
        float weight = 1.0f / num_steps_;

        for (float zf : zoom_factors) {
            // Upscale the image by the zoom factor
            cv::Mat upscaled_mat;
            cv::resize(input_mat, upscaled_mat, cv::Size(), zf, zf, cv::INTER_LINEAR);

            // Crop the center of the upscaled image to bring it back to the original size
            int crop_x = (upscaled_mat.cols - input_mat.cols) / 2;
            int crop_y = (upscaled_mat.rows - input_mat.rows) / 2;

            cv::Rect roi(crop_x, crop_y, input_mat.cols, input_mat.rows);
            cv::Mat cropped_zoomed_mat = upscaled_mat(roi);

            // Add this zoomed layer to the final blurred image
            cv::addWeighted(blurred_mat, 1.0, cropped_zoomed_mat, weight, 0, blurred_mat);
        }

        // 3. --- Clamp and Convert back to LibTorch Tensor ---
        cv::max(blurred_mat, 0, blurred_mat);
        cv::min(blurred_mat, 1, blurred_mat);

        torch::Tensor output_tensor = xt::utils::image::mat_to_tensor_local(blurred_mat);

        return output_tensor;
    }

} // namespace xt::transforms::image