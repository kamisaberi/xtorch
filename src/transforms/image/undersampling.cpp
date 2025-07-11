#include "include/transforms/image/undersampling.h"


//
// #include "include/transforms/image/undersampling.h"
// #include <vector>
// #include <stdexcept>
// #include <opencv2/opencv.hpp> // Include OpenCV headers
//
// // You should have a shared utility file for these conversions.
// // I'm assuming it's available via this include.
// #include "include/utils/conversion_utils.h"
//
// namespace xt::transforms::image {
//
//     UnderSampling::UnderSampling() = default;
//
//     UnderSampling::UnderSampling(std::vector<int64_t> crop_size, int num_samples)
//         : crop_size_(crop_size), num_samples_(num_samples) {
//
//         if (crop_size_.size() != 2) {
//             throw std::invalid_argument("UnderSampling crop_size must be a vector of two ints (height, width).");
//         }
//         if (num_samples_ <= 0) {
//             throw std::invalid_argument("UnderSampling requires num_samples to be a positive integer.");
//         }
//     }
//
//     auto UnderSampling::forward(std::initializer_list<std::any> tensors) -> std::any {
//         std::vector<std::any> any_vec(tensors);
//         if (any_vec.empty()) {
//             throw std::invalid_argument("UnderSampling::forward received an empty list of tensors.");
//         }
//         torch::Tensor img_tensor = std::any_cast<torch::Tensor>(any_vec[0]);
//
//         if (!img_tensor.defined() || img_tensor.dim() != 3) {
//             throw std::invalid_argument("UnderSampling expects a defined 3D image tensor (C, H, W).");
//         }
//
//         // 1. Convert the input LibTorch tensor to an OpenCV Mat
//         // Using the same utility function as in OverSampling.
//         cv::Mat img_mat = xt::utils::image::tensor_to_mat_local(img_tensor);
//
//         // 2. Get dimensions
//         const int img_h = img_mat.rows;
//         const int img_w = img_mat.cols;
//         const int crop_h = static_cast<int>(crop_size_[0]);
//         const int crop_w = static_cast<int>(crop_size_[1]);
//
//         if (crop_h > img_h || crop_w > img_w) {
//             throw std::invalid_argument("Crop size cannot be larger than image size.");
//         }
//
//         // 3. Create a vector to hold the final torch::Tensors
//         std::vector<torch::Tensor> cropped_tensors;
//         cropped_tensors.reserve(num_samples_);
//
//         // 4. Generate the random crops
//         for (int i = 0; i < num_samples_; ++i) {
//             // Generate a random top-left corner for the crop region
//             int top = cv::theRNG().uniform(0, img_h - crop_h + 1);
//             int left = cv::theRNG().uniform(0, img_w - crop_w + 1);
//
//             // Define the random crop rectangle
//             cv::Rect random_rect(left, top, crop_w, crop_h);
//
//             // Extract the crop from the source Mat
//             cv::Mat cropped_mat = img_mat(random_rect);
//
//             // Convert the cropped Mat back to a tensor and add it to our list
//             cropped_tensors.push_back(xt::utils::image::mat_to_tensor_local(cropped_mat));
//         }
//
//         // 5. Stack all generated crops into a single output tensor
//         return torch::stack(cropped_tensors, 0);
//     }
//
// } // namespace xt::transforms::image


namespace xt::transforms::image {

    UnderSampling::UnderSampling() = default;

    UnderSampling::UnderSampling(std::vector<int64_t> crop_size, int num_samples)
        : crop_size_(crop_size), num_samples_(num_samples) {

        if (crop_size_.size() != 2) {
            throw std::invalid_argument("UnderSampling crop_size must be a vector of two ints (height, width).");
        }
        if (num_samples_ <= 0) {
            throw std::invalid_argument("UnderSampling requires num_samples to be a positive integer.");
        }
    }

    auto UnderSampling::forward(std::initializer_list<std::any> tensors) -> std::any {
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("UnderSampling::forward received an empty list of tensors.");
        }
        torch::Tensor img_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!img_tensor.defined() || img_tensor.dim() != 3) {
            throw std::invalid_argument("UnderSampling expects a defined 3D image tensor (C, H, W).");
        }

        // 1. Convert the input LibTorch tensor to an OpenCV Mat
        // Using the same utility function as in OverSampling.
        cv::Mat img_mat = xt::utils::image::tensor_to_mat_local(img_tensor);

        // 2. Get dimensions
        const int img_h = img_mat.rows;
        const int img_w = img_mat.cols;
        const int crop_h = static_cast<int>(crop_size_[0]);
        const int crop_w = static_cast<int>(crop_size_[1]);

        if (crop_h > img_h || crop_w > img_w) {
            throw std::invalid_argument("Crop size cannot be larger than image size.");
        }

        // 3. Create a vector to hold the final torch::Tensors
        std::vector<torch::Tensor> cropped_tensors;
        cropped_tensors.reserve(num_samples_);

        // 4. Generate the random crops
        for (int i = 0; i < num_samples_; ++i) {
            // Generate a random top-left corner for the crop region
            int top = cv::theRNG().uniform(0, img_h - crop_h + 1);
            int left = cv::theRNG().uniform(0, img_w - crop_w + 1);

            // Define the random crop rectangle
            cv::Rect random_rect(left, top, crop_w, crop_h);

            // Extract the crop from the source Mat
            cv::Mat cropped_mat = img_mat(random_rect);

            // Convert the cropped Mat back to a tensor and add it to our list
            cropped_tensors.push_back(xt::utils::image::mat_to_tensor_local(cropped_mat));
        }

        // 5. Stack all generated crops into a single output tensor
        return torch::stack(cropped_tensors, 0);
    }

} // namespace xt::transforms::image