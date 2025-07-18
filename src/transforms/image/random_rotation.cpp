#include <transforms/image/random_rotation.h>


// --- Example Main (for testing) ---
// #include "transforms/image/random_rotation.h"
// #include <iostream>
// #include <opencv2/highgui.hpp>
// #include "utils/image_conversion.h"
//
// int main() {
//     // 1. Create a non-square image with an 'F' to see rotation and expansion.
//     cv::Mat image_mat(200, 300, CV_8UC3, cv::Scalar(255, 255, 255));
//     // Draw a large 'F'
//     cv::rectangle(image_mat, {50, 50}, {80, 150}, {0,0,255}, -1);
//     cv::rectangle(image_mat, {80, 50}, {180, 80}, {0,0,255}, -1);
//     cv::rectangle(image_mat, {80, 100}, {150, 120}, {0,0,255}, -1);
//
//     cv::imwrite("rotation_before.png", image_mat);
//     std::cout << "Saved rotation_before.png (Shape: " << image_mat.size() << ")" << std::endl;
//
//     torch::Tensor image = xt::utils::image::mat_to_tensor_float(image_mat);
//
//     std::cout << "--- Applying RandomRotation ---" << std::endl;
//
//     // 2. Define transform with a range of [-30, 30] degrees and expansion enabled.
//     xt::transforms::image::RandomRotation rotator(
//         /*degrees=*/{-30.0, 30.0},
//         /*expand=*/true,
//         /*fill=*/{0.5, 0.5, 0.5}
//     );
//
//     // 3. Apply the transform
//     torch::Tensor transformed_tensor = std::any_cast<torch::Tensor>(rotator.forward({image}));
//
//     // 4. Save the result
//     cv::Mat transformed_mat = xt::utils::image::tensor_to_mat_8u(transformed_tensor);
//     cv::imwrite("rotation_after.png", transformed_mat);
//     std::cout << "Saved rotation_after.png (Shape: " << transformed_mat.size() << ")" << std::endl;
//
//     return 0;
// }


namespace xt::transforms::image {

    RandomRotation::RandomRotation() : RandomRotation({-10.0, 10.0}) {}

    RandomRotation::RandomRotation(
        std::pair<double, double> degrees,
        bool expand,
        const std::vector<double>& fill,
        const std::string& interpolation)
        : degrees_(degrees), expand_(expand) {

        // --- Parameter Validation ---
        if (degrees_.first > degrees_.second) {
            throw std::invalid_argument("Degrees range must be valid (min <= max).");
        }
        if (fill.size() != 3 && fill.size() != 1) {
            throw std::invalid_argument("Fill color must be a vector of size 1 or 3.");
        }

        // --- Initialize Members ---
        if (fill.size() == 3) {
            fill_color_ = cv::Scalar(fill[0], fill[1], fill[2]);
        } else {
            fill_color_ = cv::Scalar::all(fill[0]);
        }

        if (interpolation == "bilinear") {
            interpolation_flag_ = cv::INTER_LINEAR;
        } else if (interpolation == "nearest") {
            interpolation_flag_ = cv::INTER_NEAREST;
        } else {
            throw std::invalid_argument("Unsupported interpolation type.");
        }

        std::random_device rd;
        gen_.seed(rd());
    }

    auto RandomRotation::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("RandomRotation::forward received an empty list.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to RandomRotation is not defined.");
        }

        // --- Select a random angle ---
        std::uniform_real_distribution<double> angle_dist(degrees_.first, degrees_.second);
        double angle = angle_dist(gen_);

        if (std::abs(angle) < 1e-6) {
            // No rotation needed
            return input_tensor;
        }

        // --- Convert to OpenCV Mat (Float for precision) ---
        cv::Mat input_mat_32f = xt::utils::image::tensor_to_mat_float(input_tensor);
        auto height = input_mat_32f.rows;
        auto width = input_mat_32f.cols;
        cv::Point2f center(width / 2.0f, height / 2.0f);

        // --- Get Rotation Matrix and Handle Expansion ---
        cv::Mat rotation_matrix = cv::getRotationMatrix2D(center, angle, 1.0);
        cv::Size output_size = input_mat_32f.size();

        if (expand_) {
            // Calculate the bounding box of the rotated image
            double rad_angle = angle * CV_PI / 180.0;
            double abs_cos = std::abs(cos(rad_angle));
            double abs_sin = std::abs(sin(rad_angle));

            int new_width = static_cast<int>(height * abs_sin + width * abs_cos);
            int new_height = static_cast<int>(height * abs_cos + width * abs_sin);
            output_size = cv::Size(new_width, new_height);

            // Adjust the rotation matrix to account for the new canvas size
            rotation_matrix.at<double>(0, 2) += (new_width / 2.0) - center.x;
            rotation_matrix.at<double>(1, 2) += (new_height / 2.0) - center.y;
        }

        // --- Apply the Affine Warp ---
        cv::Mat output_mat;
        cv::warpAffine(
            input_mat_32f,
            output_mat,
            rotation_matrix,
            output_size,
            interpolation_flag_,
            cv::BORDER_CONSTANT,
            fill_color_
        );

        // --- Convert back to LibTorch Tensor ---
        torch::Tensor output_tensor = xt::utils::image::mat_to_tensor_float(output_mat);

        return output_tensor;
    }

} // namespace xt::transforms::image