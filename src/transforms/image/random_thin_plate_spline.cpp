#include "include/transforms/image/random_thin_plate_spline.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/shape.hpp> // For ThinPlateSplineShapeTransformer

// --- Example Main (for testing) ---
// #include "transforms/image/random_thin_plate_spline.h"
// #include <iostream>
// #include <opencv2/highgui.hpp>
// #include "utils/image_conversion.h"
//
// int main() {
//     // 1. Create a sample grid image to visualize the warp clearly.
//     cv::Mat image_mat(256, 256, CV_8UC3, cv::Scalar(255, 255, 255));
//     for (int i = 0; i < image_mat.rows; i += 16) cv::line(image_mat, {0, i}, {image_mat.cols, i}, {0, 0, 0}, 1);
//     for (int i = 0; i < image_mat.cols; i += 16) cv::line(image_mat, {i, 0}, {i, image_mat.rows}, {0, 0, 0}, 1);
//     cv::imwrite("tps_before.png", image_mat);
//     std::cout << "Saved tps_before.png" << std::endl;
//
//     torch::Tensor image = xt::utils::image::mat_to_tensor_float(image_mat);
//
//     std::cout << "--- Applying RandomThinPlateSpline ---" << std::endl;
//
//     // 2. Define transform with a 5x5 grid and strong distortion.
//     xt::transforms::image::RandomThinPlateSpline warper(
//         /*grid_size=*/5,
//         /*distortion_scale=*/0.4,
//         /*p=*/1.0,
//         /*fill=*/{0.5, 0.5, 0.5} // Fill with gray
//     );
//
//     // 3. Apply the transform
//     torch::Tensor transformed_tensor = std::any_cast<torch::Tensor>(warper.forward({image}));
//
//     // 4. Save the result
//     cv::Mat transformed_mat = xt::utils::image::tensor_to_mat_8u(transformed_tensor);
//     cv::imwrite("tps_after.png", transformed_mat);
//     std::cout << "Saved tps_after.png" << std::endl;
//
//     return 0;
// }


namespace xt::transforms::image {

    RandomThinPlateSpline::RandomThinPlateSpline() : RandomThinPlateSpline(4, 0.2, 0.5) {}

    RandomThinPlateSpline::RandomThinPlateSpline(
        int grid_size,
        double distortion_scale,
        double p,
        const std::vector<double>& fill,
        const std::string& interpolation)
        : grid_size_(grid_size), distortion_scale_(distortion_scale), p_(p) {

        // --- Parameter Validation ---
        if (grid_size_ < 2) {
            throw std::invalid_argument("Grid size must be at least 2.");
        }
        if (distortion_scale_ < 0.0) {
            throw std::invalid_argument("Distortion scale must be non-negative.");
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

    auto RandomThinPlateSpline::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- Probability Check ---
        std::uniform_real_distribution<> prob_dist(0.0, 1.0);
        if (prob_dist(gen_) >= p_) {
            return tensors.begin()[0]; // Return the original tensor
        }

        // --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("RandomThinPlateSpline::forward received an empty list.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to RandomThinPlateSpline is not defined.");
        }

        // --- Convert to OpenCV Mat (Float for precision) ---
        cv::Mat input_mat_32f = xt::utils::image::tensor_to_mat_float(input_tensor);
        auto h = input_mat_32f.rows;
        auto w = input_mat_32f.cols;

        // --- Generate Control Points ---
        // 1. Create a regular grid of source points
        std::vector<cv::Point2f> src_points;
        for (int i = 0; i < grid_size_; ++i) {
            for (int j = 0; j < grid_size_; ++j) {
                float x = (float)j * (w - 1) / (grid_size_ - 1);
                float y = (float)i * (h - 1) / (grid_size_ - 1);
                src_points.emplace_back(x, y);
            }
        }

        // 2. Create destination points by randomly perturbing the source points
        std::vector<cv::Point2f> dst_points = src_points;
        float max_dx = (float)w / (grid_size_ - 1) * distortion_scale_;
        float max_dy = (float)h / (grid_size_ - 1) * distortion_scale_;
        std::uniform_real_distribution<float> x_dist(-max_dx, max_dx);
        std::uniform_real_distribution<float> y_dist(-max_dy, max_dy);

        for (auto& pt : dst_points) {
            pt.x += x_dist(gen_);
            pt.y += y_dist(gen_);
        }

        // --- Compute and Apply the Warp ---
        // 1. Create the TPS transformer
        cv::Ptr<cv::ThinPlateSplineShapeTransformer> tps = cv::createThinPlateSplineShapeTransformer();

        // 2. Estimate the transformation
        std::vector<cv::DMatch> matches;
        for(size_t i = 0; i < src_points.size(); ++i) {
            matches.emplace_back(i, i, 0);
        }
        tps->estimateTransformation(dst_points, src_points, matches);

        // 3. Warp the image
        cv::Mat output_mat;
        tps->warpImage(
            input_mat_32f,
            output_mat,
            interpolation_flag_,
            cv::BORDER_CONSTANT,
            fill_color_
        );

        // --- Convert back to LibTorch Tensor ---
        torch::Tensor output_tensor = xt::utils::image::mat_to_tensor_float(output_mat);

        return output_tensor;
    }

} // namespace xt::transforms::image