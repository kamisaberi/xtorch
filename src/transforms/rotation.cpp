#include "../../include/transforms/rotation.h"

namespace xt::data::transforms {

    Rotation::Rotation(double angle_deg) : angle(angle_deg) {
    }

    torch::Tensor Rotation::operator()(const torch::Tensor &input_tensor) {
        // Convert torch::Tensor to OpenCV Mat (assuming CHW format and float32 in [0,1])
        auto img_tensor = input_tensor.detach().cpu().clone();
        img_tensor = img_tensor.permute({1, 2, 0}); // Convert CHW -> HWC
        img_tensor = img_tensor.mul(255).clamp(0, 255).to(torch::kU8);

        cv::Mat img(img_tensor.size(0), img_tensor.size(1), CV_8UC3);
        std::memcpy((void *) img.data, img_tensor.data_ptr(), sizeof(uint8_t) * img_tensor.numel());

        // Compute center of the image and get rotation matrix
        cv::Point2f center(img.cols / 2.0f, img.rows / 2.0f);
        cv::Mat rot_matrix = cv::getRotationMatrix2D(center, angle, 1.0);

        // Rotate the image
        cv::Mat rotated_img;
        cv::warpAffine(img, rotated_img, rot_matrix, img.size(), cv::INTER_LINEAR);

        // Convert back to Tensor
        torch::Tensor rotated_tensor = torch::from_blob(
            rotated_img.data,
            {rotated_img.rows, rotated_img.cols, 3},
            torch::kUInt8).clone();

        rotated_tensor = rotated_tensor.permute({2, 0, 1}); // HWC -> CHW
        rotated_tensor = rotated_tensor.to(torch::kFloat32).div(255); // Normalize to [0,1]

        return rotated_tensor;
    }




}