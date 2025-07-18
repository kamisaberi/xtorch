#include <transforms/image/blur.h>

namespace xt::transforms::image {

    Blur::Blur() : kernel_size_({3, 3}), sigma_({0.0, 0.0}) {}

    Blur::Blur(std::vector<int64_t> kernel_size, std::vector<double> sigma)
        : kernel_size_(kernel_size), sigma_(sigma) {

        if (kernel_size_.size() != 2) {
            throw std::invalid_argument("Blur kernel_size must be a vector of two integers.");
        }
        if (sigma_.size() != 2) {
            throw std::invalid_argument("Blur sigma must be a vector of two doubles.");
        }
        if (kernel_size_[0] <= 0 || kernel_size_[1] <= 0) {
            throw std::invalid_argument("Kernel dimensions must be positive.");
        }
        if (kernel_size_[0] % 2 == 0 || kernel_size_[1] % 2 == 0) {
            throw std::invalid_argument("Kernel dimensions must be odd.");
        }
    }

    auto Blur::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("Blur::forward received an empty list of tensors.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to Blur is not defined.");
        }

        // 2. --- Convert to OpenCV Mat ---
        // We use your existing conversion utility.
        cv::Mat input_mat = xt::utils::image::tensor_to_mat_local(input_tensor);

        // 3. --- Apply Gaussian Blur ---
        cv::Mat blurred_mat;
        cv::Size kernel(static_cast<int>(kernel_size_[1]), static_cast<int>(kernel_size_[0]));

        cv::GaussianBlur(
            input_mat,      // source image
            blurred_mat,    // destination image
            kernel,         // kernel size (width, height)
            sigma_[0],      // sigmaX
            sigma_[1]       // sigmaY
        );

        // 4. --- Convert back to LibTorch Tensor ---
        torch::Tensor output_tensor = xt::utils::image::mat_to_tensor_local(blurred_mat);

        return output_tensor;
    }

} // namespace xt::transforms::image