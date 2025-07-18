#include <transforms/image/emboss.h>

// #include "transforms/image/emboss.h"
// #include <iostream>
//
// int main() {
//     // 1. Create a dummy color image tensor
//     torch::Tensor image = torch::rand({3, 200, 200});
//
//     // 2. Instantiate the transform for a moderate, blended effect
//     // 60% of the final image will be the emboss effect.
//     // The effect itself will be at 1.5x strength.
//     xt::transforms::image::Emboss embosser(0.6f, 1.5f);
//
//     // 3. Apply the transform
//     std::any result_any = embosser.forward({image});
//     torch::Tensor embossed_image = std::any_cast<torch::Tensor>(result_any);
//
//     // 4. Check the output
//     std::cout << "Original image shape: " << image.sizes() << std::endl;
//     std::cout << "Embossed image shape: " << embossed_image.sizes() << std::endl; // Should be the same
//
//     // You can save the result to visually inspect it.
//     // cv::Mat output_mat = xt::utils::image::tensor_to_mat_8u(embossed_image);
//     // cv::imwrite("embossed_image_v2.png", output_mat);
//
//     return 0;
// }

namespace xt::transforms::image {

    Emboss::Emboss() : alpha_(0.5f), strength_(0.5f) {}

    Emboss::Emboss(float alpha, float strength) : alpha_(alpha), strength_(strength) {
        if (alpha < 0.0f || alpha > 1.0f) {
            throw std::invalid_argument("Emboss alpha must be between 0.0 and 1.0.");
        }
    }

    auto Emboss::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("Emboss::forward received an empty list of tensors.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to Emboss is not defined.");
        }

        // 2. --- Convert to OpenCV Mat ---
        // We will work with a grayscale version for the classic emboss effect.
        torch::Tensor gray_tensor;
        if (input_tensor.size(0) == 3) {
            auto weights = torch::tensor({0.299, 0.587, 0.114}, input_tensor.options()).view({3, 1, 1});
            gray_tensor = (input_tensor * weights).sum(0, true);
        } else {
            gray_tensor = input_tensor;
        }

        // Convert to an 8-bit Mat for filtering
        cv::Mat gray_mat_8u = xt::utils::image::tensor_to_mat_8u(gray_tensor);

        // 3. --- Define the Emboss Kernel and Apply Filter ---
        cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
            -2, -1,  0,
            -1,  1,  1,
             0,  1,  2);

        kernel *= strength_; // Apply the strength factor

        cv::Mat embossed_mat;
        cv::filter2D(gray_mat_8u, embossed_mat, -1, kernel);

        // The result of the filter is often shifted. A common practice is to add
        // 128 to bring the mid-tones back to gray.
        embossed_mat += 128;

        // 4. --- Blend with Original and Convert Back ---
        cv::Mat final_mat;
        // The original image for blending should also be grayscale
        cv::addWeighted(gray_mat_8u, 1.0 - alpha_, embossed_mat, alpha_, 0.0, final_mat);

        // Convert the 8-bit result back to a float tensor [0, 1]
        torch::Tensor output_tensor = xt::utils::image::mat_to_tensor_float(final_mat);

        // If the original was a color image, we can broadcast the grayscale result
        // back to 3 channels to maintain the original shape.
        if (input_tensor.size(0) == 3) {
            output_tensor = output_tensor.repeat({3, 1, 1});
        }

        return output_tensor;
    }

} // namespace xt::transforms::image