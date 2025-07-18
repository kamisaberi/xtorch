#include <transforms/image/median_blur.h>

//
// #include "transforms/image/median_blur.h"
// #include <iostream>
//
// // Helper function to add salt-and-pepper noise for the demo
// torch::Tensor add_salt_pepper_noise(const torch::Tensor& image, float amount) {
//     auto noisy_image = image.clone();
//     int64_t num_pixels = image.numel() / image.size(0); // num pixels per channel
//     int64_t num_salt = static_cast<int64_t>(num_pixels * amount / 2.0);
//     int64_t num_pepper = static_cast<int64_t>(num_pixels * amount / 2.0);
//
//     for (int c = 0; c < image.size(0); ++c) {
//         // Salt
//         auto salt_coords_y = torch::randint(0, image.size(1), {num_salt});
//         auto salt_coords_x = torch::randint(0, image.size(2), {num_salt});
//         noisy_image.index_put_({c, salt_coords_y, salt_coords_x}, 1.0);
//
//         // Pepper
//         auto pepper_coords_y = torch::randint(0, image.size(1), {num_pepper});
//         auto pepper_coords_x = torch::randint(0, image.size(2), {num_pepper});
//         noisy_image.index_put_({c, pepper_coords_y, pepper_coords_x}, 0.0);
//     }
//     return noisy_image;
// }
//
// int main() {
//     // 1. Create a dummy image and add salt-and-pepper noise
//     torch::Tensor image = torch::ones({3, 200, 200}) * 0.5; // Solid gray
//     torch::Tensor noisy_image = add_salt_pepper_noise(image, 0.1); // Add 10% noise
//
//     // 2. Instantiate the transform with a 5x5 kernel
//     xt::transforms::image::MedianBlur blurer(5);
//
//     // 3. Apply the transform to the noisy image
//     std::any result_any = blurer.forward({noisy_image});
//     torch::Tensor denoised_image = std::any_cast<torch::Tensor>(result_any);
//
//     // 4. Check the output
//     std::cout << "Original image shape: " << image.sizes() << std::endl;
//     std::cout << "Denoised image shape: " << denoised_image.sizes() << std::endl;
//
//     // The mean of the denoised image should be very close to the original 0.5,
//     // as the filter should have removed most of the salt (1.0) and pepper (0.0) pixels.
//     std::cout << "Mean of original image: " << image.mean().item<float>() << std::endl;
//     std::cout << "Mean of noisy image: " << noisy_image.mean().item<float>() << std::endl;
//     std::cout << "Mean of denoised image: " << denoised_image.mean().item<float>() << std::endl;
//
//     // You could save the noisy and denoised images to see the effect.
//     // cv::Mat noisy_mat = xt::utils::image::tensor_to_mat_8u(noisy_image);
//     // cv::imwrite("noisy_image.png", noisy_mat);
//     // cv::Mat denoised_mat = xt::utils::image::tensor_to_mat_8u(denoised_image);
//     // cv::imwrite("denoised_image.png", denoised_mat);
//
//     return 0;
// }

namespace xt::transforms::image {

    MedianBlur::MedianBlur() : kernel_size_(3) {}

    MedianBlur::MedianBlur(int kernel_size) : kernel_size_(kernel_size) {
        if (kernel_size_ <= 1 || kernel_size_ % 2 == 0) {
            throw std::invalid_argument("MedianBlur kernel_size must be an odd integer greater than 1.");
        }
    }

    auto MedianBlur::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("MedianBlur::forward received an empty list of tensors.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to MedianBlur is not defined.");
        }

        // 2. --- Convert to OpenCV Mat ---
        // medianBlur requires an 8-bit integer Mat (CV_8U).
        cv::Mat input_mat_8u = xt::utils::image::tensor_to_mat_8u(input_tensor);

        // 3. --- Apply Median Blur ---
        cv::Mat blurred_mat;
        cv::medianBlur(
            input_mat_8u,   // source image
            blurred_mat,    // destination image
            kernel_size_    // aperture linear size; must be odd and greater than 1
        );

        // 4. --- Convert back to LibTorch Tensor ---
        // Convert the 8-bit result back to a float tensor in the [0, 1] range.
        torch::Tensor output_tensor = xt::utils::image::mat_to_tensor_float(blurred_mat);

        return output_tensor;
    }

} // namespace xt::transforms::image