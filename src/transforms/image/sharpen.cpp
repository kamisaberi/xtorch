#include "include/transforms/image/sharpen.h"

// #include "transforms/image/sharpen.h"
// #include <iostream>
//
// int main() {
//     // 1. Create a dummy image tensor. A slightly blurred image will show the effect best.
//     torch::Tensor image = torch::rand({3, 200, 200});
//     // Let's simulate a slightly soft image.
//     // (This would require a blur transform, but we can just use the random one for demo)
//
//     // 2. Instantiate the transform for a strong sharpening effect.
//     // alpha=0.8 means the final image is 80% sharpened effect.
//     // lightness=1.5 means the sharpening itself is intense.
//     xt::transforms::image::Sharpen sharpener(0.8f, 1.5f);
//
//     // 3. Apply the transform
//     std::any result_any = sharpener.forward({image});
//     torch::Tensor sharpened_image = std::any_cast<torch::Tensor>(result_any);
//
//     // 4. Check the output
//     std::cout << "Original image shape: " << image.sizes() << std::endl;
//     std::cout << "Sharpened image shape: " << sharpened_image.sizes() << std::endl;
//
//     // The standard deviation of the sharpened image should be higher than the original,
//     // indicating increased contrast at the edges.
//     std::cout << "Original std dev: " << image.std().item<float>() << std::endl;
//     std::cout << "Sharpened std dev: " << sharpened_image.std().item<float>() << std::endl;
//
//     // You could save the output image to see the effect.
//     // cv::Mat output_mat = xt::utils::image::tensor_to_mat_8u(sharpened_image);
//     // cv::imwrite("sharpened_image.png", output_mat);
//
//     return 0;
// }

namespace xt::transforms::image {

    Sharpen::Sharpen() : alpha_(0.5f), lightness_(0.5f) {}

    Sharpen::Sharpen(float alpha, float lightness) : alpha_(alpha), lightness_(lightness) {
        if (alpha < 0.0f || alpha > 1.0f) {
            throw std::invalid_argument("Sharpen alpha (blending factor) must be between 0.0 and 1.0.");
        }
        if (lightness < 0.0f) {
            throw std::invalid_argument("Sharpen lightness must be non-negative.");
        }
    }

    auto Sharpen::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation and Conversion to Mat ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("Sharpen::forward received an empty list of tensors.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to Sharpen is not defined.");
        }

        // We will work with a float Mat [0, 1] to preserve precision.
        cv::Mat input_mat = xt::utils::image::tensor_to_mat_local(input_tensor);

        // 2. --- Define the Sharpening Kernel ---
        // A standard 3x3 sharpening kernel
        cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
             0,         -1,          0,
            -1,          5,         -1,
             0,         -1,          0);

        // The "lightness" factor adjusts the center value, controlling intensity.
        // A value of 1.0 corresponds to the standard kernel above (4*lightness + 1 = 5).
        kernel.at<float>(1, 1) = 4 * lightness_ + 1;
        kernel.row(0).col(1) *= lightness_;
        kernel.row(1).col(0) *= lightness_;
        kernel.row(1).col(2) *= lightness_;
        kernel.row(2).col(1) *= lightness_;

        // 3. --- Apply the Filter ---
        cv::Mat sharpened_effect;
        cv::filter2D(input_mat, sharpened_effect, -1, kernel);

        // 4. --- Blend with Original and Convert Back ---
        cv::Mat final_mat;
        cv::addWeighted(input_mat, 1.0 - alpha_, sharpened_effect, alpha_, 0.0, final_mat);

        // Clamp values to the valid [0, 1] range after blending
        cv::max(final_mat, 0, final_mat);
        cv::min(final_mat, 1, final_mat);

        // 5. --- Convert the final Mat back to a float tensor ---
        torch::Tensor output_tensor = xt::utils::image::mat_to_tensor_local(final_mat);

        return output_tensor;
    }

} // namespace xt::transforms::image