#include "include/transforms/image/oversampling.h"
#include <vector>
#include <stdexcept>
#include <opencv2/opencv.hpp> // Include OpenCV headers

// Assuming you have a helper file like this for conversions
// #include "include/utils/conversion_utils.h"

namespace xt::transforms::image
{
    OverSampling::OverSampling() = default;

    OverSampling::OverSampling(std::vector<int64_t> crop_size) : crop_size(crop_size)
    {
        if (crop_size.size() != 2)
        {
            throw std::invalid_argument("OverSampling crop_size must be a vector of two ints (height, width).");
        }
    }

    // Helper function to convert a single-image tensor to a cv::Mat
    // This should be in a shared utility file, but is included here for completeness.
    // cv::Mat tensor_to_mat_local(torch::Tensor tensor) {
    //     tensor = tensor.squeeze().detach().clone();
    //     tensor = tensor.permute({1, 2, 0}); // CHW -> HWC
    //
    //     // IMPORTANT: We assume the tensor is float and needs to be handled as such.
    //     // OpenCV can work with float Mats directly (CV_32FCn).
    //     cv::Mat mat(tensor.size(0), tensor.size(1), CV_32FC(tensor.size(2)));
    //     std::memcpy(mat.data, tensor.data_ptr<float>(), sizeof(float) * tensor.numel());
    //     return mat;
    // }

    // // Helper function to convert a cv::Mat back to a tensor.
    // torch::Tensor mat_to_tensor_local(const cv::Mat& mat) {
    //     // Ensure mat is float type as expected
    //     cv::Mat mat_float;
    //     if (mat.type() != CV_32FC3 && mat.type() != CV_32FC1) {
    //          mat.convertTo(mat_float, CV_32F);
    //     } else {
    //         mat_float = mat;
    //     }
    //
    //     cv::Mat mat_cont = mat_float.isContinuous() ? mat_float : mat_float.clone();
    //
    //     torch::Tensor tensor = torch::from_blob(mat_cont.data, {mat_cont.rows, mat_cont.cols, mat_cont.channels()}, torch::kFloat32);
    //     tensor = tensor.permute({2, 0, 1}); // HWC -> CHW
    //     return tensor.clone(); // Clone to make sure the tensor owns its memory
    // }


    auto OverSampling::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty())
        {
            throw std::invalid_argument("OverSampling::forward received an empty list of tensors.");
        }
        torch::Tensor img_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!img_tensor.defined() || img_tensor.dim() != 3)
        {
            throw std::invalid_argument("OverSampling expects a defined 3D image tensor (C, H, W).");
        }

        // 1. Convert the input LibTorch tensor to an OpenCV Mat
        cv::Mat img_mat = xt::utils::image::tensor_to_mat_local(img_tensor);

        // 2. Get dimensions
        const int img_h = img_mat.rows;
        const int img_w = img_mat.cols;
        const int crop_h = static_cast<int>(crop_size[0]);
        const int crop_w = static_cast<int>(crop_size[1]);

        if (crop_h > img_h || crop_w > img_w)
        {
            throw std::invalid_argument("Crop size cannot be larger than image size.");
        }

        // 3. Define the 5 crop regions using cv::Rect
        std::vector<cv::Rect> crop_rects;
        crop_rects.push_back(cv::Rect(0, 0, crop_w, crop_h)); // Top-Left
        crop_rects.push_back(cv::Rect(img_w - crop_w, 0, crop_w, crop_h)); // Top-Right
        crop_rects.push_back(cv::Rect(0, img_h - crop_h, crop_w, crop_h)); // Bottom-Left
        crop_rects.push_back(cv::Rect(img_w - crop_w, img_h - crop_h, crop_w, crop_h)); // Bottom-Right
        crop_rects.push_back(cv::Rect((img_w - crop_w) / 2, (img_h - crop_h) / 2, crop_w, crop_h)); // Center

        // 4. Create a vector to hold the final torch::Tensors
        std::vector<torch::Tensor> augmented_tensors;
        augmented_tensors.reserve(10);

        // 5. Generate the 10 crops
        for (const auto& rect : crop_rects)
        {
            // Extract the crop
            cv::Mat cropped_mat = img_mat(rect);

            // Generate the flipped version
            cv::Mat flipped_mat;
            cv::flip(cropped_mat, flipped_mat, 1); // 1 for horizontal flip

            // Convert both back to tensors and add to the list
            augmented_tensors.push_back(xt::utils::image::mat_to_tensor_local(cropped_mat));
            augmented_tensors.push_back(xt::utils::image::mat_to_tensor_local(flipped_mat));
        }

        // 6. Stack all 10 tensors into a single output tensor
        return torch::stack(augmented_tensors, 0);
    }
} // namespace xt::transforms::image
