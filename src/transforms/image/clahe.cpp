#include "include/transforms/image/clahe.h"



// #include "transforms/image/clahe.h"
// #include <iostream>
//
// int main() {
//     // 1. Load an image that could benefit from CLAHE (e.g., one with dark shadows)
//     // For this example, we'll create a dummy one.
//     torch::Tensor image = torch::ones({3, 200, 200}) * 0.1; // Dark image
//     // Add a slightly brighter, low-contrast area
//     image.slice(1, 50, 150).slice(2, 50, 150) = 0.2;
//
//     std::cout << "Original Image Min/Max: " << image.min().item<float>()
//               << " / " << image.max().item<float>() << std::endl;
//
//     // 2. Instantiate the transform with a moderate clip limit
//     xt::transforms::image::CLAHE enhancer(4.0, {8, 8});
//
//     // 3. Apply the transform
//     std::any result_any = enhancer.forward({image});
//     torch::Tensor enhanced_image = std::any_cast<torch::Tensor>(result_any);
//
//     // 4. Check the output
//     std::cout << "Enhanced Image Min/Max: " << enhanced_image.min().item<float>()
//               << " / " << enhanced_image.max().item<float>() << std::endl;
//     std::cout << "Image shape remains: " << enhanced_image.sizes() << std::endl;
//
//     // The min/max range of the enhanced image will be much wider, indicating
//     // that the local contrast has been successfully increased.
//
//     return 0;
// }

namespace xt::transforms::image {

    // --- Constructor & Destructor ---
    CLAHE::CLAHE() {
        clahe_ptr_ = cv::createCLAHE(2.0, cv::Size(8, 8));
    }

    CLAHE::CLAHE(double clip_limit, std::vector<int> tile_grid_size) {
        if (tile_grid_size.size() != 2) {
            throw std::invalid_argument("CLAHE tile_grid_size must be a vector of two integers.");
        }
        cv::Size grid_size(tile_grid_size[1], tile_grid_size[0]);
        clahe_ptr_ = cv::createCLAHE(clip_limit, grid_size);
    }

    // Destructor must be defined in the .cpp file where cv::CLAHE is a complete type.
    CLAHE::~CLAHE() = default;

    // --- Rule of 5 Implementation for PIMPL ---
    CLAHE::CLAHE(const CLAHE& other) {
        // Recreate the CLAHE object with the same parameters
        double clip_limit = other.clahe_ptr_->getClipLimit();
        cv::Size grid_size = other.clahe_ptr_->getTilesGridSize();
        clahe_ptr_ = cv::createCLAHE(clip_limit, grid_size);
    }

    CLAHE::CLAHE(CLAHE&& other) noexcept : clahe_ptr_(std::move(other.clahe_ptr_)) {}

    CLAHE& CLAHE::operator=(const CLAHE& other) {
        if (this != &other) {
            double clip_limit = other.clahe_ptr_->getClipLimit();
            cv::Size grid_size = other.clahe_ptr_->getTilesGridSize();
            clahe_ptr_ = cv::createCLAHE(clip_limit, grid_size);
        }
        return *this;
    }

    CLAHE& CLAHE::operator=(CLAHE&& other) noexcept {
        if (this != &other) {
            clahe_ptr_ = std::move(other.clahe_ptr_);
        }
        return *this;
    }


    auto CLAHE::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("CLAHE::forward received an empty list of tensors.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to CLAHE is not defined.");
        }

        int64_t num_channels = input_tensor.size(0);
        if (input_tensor.dim() != 3 || (num_channels != 1 && num_channels != 3)) {
            throw std::invalid_argument("CLAHE expects a 1-channel or 3-channel image tensor.");
        }

        // 2. --- Convert to OpenCV Mat (8-bit) ---
        // CLAHE operates on 8-bit images (0-255 range).
        cv::Mat input_mat_8u = xt::utils::image::tensor_to_mat_8u(input_tensor);

        // 3. --- Apply CLAHE ---
        cv::Mat clahe_mat;
        if (num_channels == 3) {
            // For color images, it's best to apply CLAHE only to the luminance channel (L)
            // in the Lab color space to avoid distorting colors.
            cv::Mat lab_image;
            cv::cvtColor(input_mat_8u, lab_image, cv::COLOR_BGR2Lab);

            std::vector<cv::Mat> lab_planes(3);
            cv::split(lab_image, lab_planes); // L, a, b

            // Apply CLAHE to the L-channel
            clahe_ptr_->apply(lab_planes[0], lab_planes[0]);

            cv::merge(lab_planes, lab_image);
            cv::cvtColor(lab_image, clahe_mat, cv::COLOR_Lab2BGR);

        } else { // num_channels == 1
            // For grayscale images, apply directly.
            clahe_ptr_->apply(input_mat_8u, clahe_mat);
        }

        // 4. --- Convert back to LibTorch Tensor (Float) ---
        // Convert the 8-bit result back to a float tensor in the [0, 1] range.
        torch::Tensor output_tensor = xt::utils::image::mat_to_tensor_float(clahe_mat);

        return output_tensor;
    }

} // namespace xt::transforms::image