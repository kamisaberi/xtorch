#include "include/transforms/image/random_perspective.h"


// --- Example Main (for testing) ---
// #include "transforms/image/random_perspective.h"
// #include <iostream>
// #include <opencv2/highgui.hpp>
// #include "utils/image_conversion.h"
//
// int main() {
//     // 1. Create a sample grid image to visualize the transformation clearly.
//     cv::Mat image_mat(256, 256, CV_8UC3, cv::Scalar(255, 255, 255));
//     for (int i = 0; i < image_mat.rows; i += 16) cv::line(image_mat, {0, i}, {image_mat.cols, i}, {0, 0, 0}, 1);
//     for (int i = 0; i < image_mat.cols; i += 16) cv::line(image_mat, {i, 0}, {i, image_mat.rows}, {0, 0, 0}, 1);
//     cv::imwrite("perspective_before.png", image_mat);
//     std::cout << "Saved perspective_before.png" << std::endl;
//
//     torch::Tensor image = xt::utils::image::mat_to_tensor_float(image_mat);
//
//     std::cout << "--- Applying RandomPerspective ---" << std::endl;
//
//     // 2. Define transform with a moderate distortion scale of 0.6.
//     //    Use p=1.0 to guarantee it runs.
//     xt::transforms::image::RandomPerspective transformer(
//         /*distortion_scale=*/0.6,
//         /*p=*/1.0,
//         /*fill=*/{0.5, 0.5, 0.5} // Fill with gray
//     );
//
//     // 3. Apply the transform
//     torch::Tensor transformed_tensor = std::any_cast<torch::Tensor>(transformer.forward({image}));
//
//     // 4. Save the result
//     cv::Mat transformed_mat = xt::utils::image::tensor_to_mat_8u(transformed_tensor);
//     cv::imwrite("perspective_after.png", transformed_mat);
//     std::cout << "Saved perspective_after.png" << std::endl;
//
//     return 0;
// }


namespace xt::transforms::image {

    RandomPerspective::RandomPerspective() : RandomPerspective(0.5, 0.5) {}

    RandomPerspective::RandomPerspective(
        double distortion_scale,
        double p,
        const std::vector<double>& fill,
        const std::string& interpolation)
        : distortion_scale_(distortion_scale), p_(p) {

        // --- Parameter Validation ---
        if (distortion_scale_ < 0.0 || distortion_scale_ > 1.0) {
            throw std::invalid_argument("Distortion scale must be between 0.0 and 1.0.");
        }
        if (p_ < 0.0 || p_ > 1.0) {
            throw std::invalid_argument("Probability p must be between 0.0 and 1.0.");
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

    auto RandomPerspective::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- Probability Check ---
        std::uniform_real_distribution<> prob_dist(0.0, 1.0);
        if (prob_dist(gen_) >= p_) {
            return tensors.begin()[0]; // Return the original tensor
        }

        // --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("RandomPerspective::forward received an empty list.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to RandomPerspective is not defined.");
        }

        // --- Convert to OpenCV Mat (Float for precision) ---
        cv::Mat input_mat_32f = xt::utils::image::tensor_to_mat_float(input_tensor);
        auto height = input_mat_32f.rows;
        auto width = input_mat_32f.cols;

        // --- Generate Random Transformation Points ---
        // 1. Define the source corners (the original image corners)
        std::vector<cv::Point2f> src_points = {
            {0.0f, 0.0f},          // top-left
            {(float)width, 0.0f},  // top-right
            {(float)width, (float)height}, // bottom-right
            {0.0f, (float)height}  // bottom-left
        };

        // 2. Define the destination corners by randomly perturbing the source corners
        std::vector<cv::Point2f> dst_points(4);
        float max_dx = (float)width * distortion_scale_ / 2.0f;
        float max_dy = (float)height * distortion_scale_ / 2.0f;
        std::uniform_real_distribution<float> x_dist(-max_dx, max_dx);
        std::uniform_real_distribution<float> y_dist(-max_dy, max_dy);

        for (int i = 0; i < 4; ++i) {
            dst_points[i] = {src_points[i].x + x_dist(gen_), src_points[i].y + y_dist(gen_)};
        }

        // --- Get Transformation Matrix and Apply Warp ---
        cv::Mat perspective_matrix = cv::getPerspectiveTransform(src_points, dst_points);

        cv::Mat output_mat;
        cv::warpPerspective(
            input_mat_32f,
            output_mat,
            perspective_matrix,
            input_mat_32f.size(),
            interpolation_flag_,
            cv::BORDER_CONSTANT,
            fill_color_
        );

        // --- Convert back to LibTorch Tensor ---
        torch::Tensor output_tensor = xt::utils::image::mat_to_tensor_float(output_mat);

        return output_tensor;
    }

} // namespace xt::transforms::image