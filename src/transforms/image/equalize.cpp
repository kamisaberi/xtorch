#include <transforms/image/equalize.h>



// #include "transforms/image/equalize.h"
// #include <iostream>
//
// int main() {
//     // 1. Create a dummy low-contrast image.
//     // Let's create a grayscale image with pixel values only in the 0.3 to 0.6 range.
//     torch::Tensor low_contrast_image = torch::rand({1, 200, 200}) * 0.3 + 0.3;
//
//     std::cout << "Original Image Min/Max: " << low_contrast_image.min().item<float>()
//               << " / " << low_contrast_image.max().item<float>() << std::endl;
//
//     // 2. Instantiate the transform
//     xt::transforms::image::Equalize equalizer;
//
//     // 3. Apply the transform
//     std::any result_any = equalizer.forward({low_contrast_image});
//     torch::Tensor equalized_image = std::any_cast<torch::Tensor>(result_any);
//
//     // 4. Check the output
//     std::cout << "Equalized Image Min/Max: " << equalized_image.min().item<float>()
//               << " / " << equalized_image.max().item<float>() << std::endl;
//     std::cout << "Image shape remains: " << equalized_image.sizes() << std::endl;
//
//     // The min/max range of the equalized image will be spread out across the
//     // full [0, 1] range, indicating that the contrast has been successfully increased.
//
//     // You could save the output to see the effect visually.
//     // cv::Mat output_mat = xt::utils::image::tensor_to_mat_8u(equalized_image);
//     // cv::imwrite("equalized_image.png", output_mat);
//
//     return 0;
// }

namespace xt::transforms::image {

    Equalize::Equalize() = default;

    auto Equalize::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation and Conversion to Mat ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("Equalize::forward received an empty list of tensors.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to Equalize is not defined.");
        }

        // Convert the input tensor to an 8-bit OpenCV Mat for processing.
        cv::Mat input_mat_8u = xt::utils::image::tensor_to_mat_8u(input_tensor);

        // 2. --- Apply Histogram Equalization ---
        cv::Mat equalized_mat;

        if (input_mat_8u.channels() == 3) {
            // For color images, equalizing the R, G, and B channels independently
            // can lead to significant and unpleasant color shifts.
            // The standard best practice is to convert to a color space with a
            // separate luminance channel (like YCbCr or HSV), equalize only that
            // channel, and then convert back.

            cv::Mat ycrcb_image;
            cv::cvtColor(input_mat_8u, ycrcb_image, cv::COLOR_BGR2YCrCb);

            std::vector<cv::Mat> ycrcb_planes;
            cv::split(ycrcb_image, ycrcb_planes);

            // Equalize the Y (luminance) channel.
            cv::equalizeHist(ycrcb_planes[0], ycrcb_planes[0]);

            // Merge the channels back and convert back to BGR.
            cv::merge(ycrcb_planes, ycrcb_image);
            cv::cvtColor(ycrcb_image, equalized_mat, cv::COLOR_YCrCb2BGR);

        } else if (input_mat_8u.channels() == 1) {
            // For grayscale images, we can apply the function directly.
            cv::equalizeHist(input_mat_8u, equalized_mat);
        } else {
            throw std::invalid_argument("Equalize transform expects a 1-channel or 3-channel image.");
        }

        // 3. --- Convert the final Mat back to a float tensor ---
        torch::Tensor output_tensor = xt::utils::image::mat_to_tensor_float(equalized_mat);

        return output_tensor;
    }

} // namespace xt::transforms::image