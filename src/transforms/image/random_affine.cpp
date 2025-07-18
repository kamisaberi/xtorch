#include <transforms/image/random_affine.h>


// --- Example Main (for testing) ---
// #include "transforms/image/random_affine.h"
// #include <iostream>
// #include <opencv2/highgui.hpp>
// #include "utils/image_conversion.h"
//
// int main() {
//     // 1. Load a sample image.
//     cv::Mat image_mat = cv::imread("sample_image.png", cv::IMREAD_COLOR);
//     if (image_mat.empty()) {
//         std::cerr << "Error: Could not load sample_image.png. Creating a dummy grid." << std::endl;
//         image_mat = cv::Mat(256, 256, CV_8UC3, cv::Scalar(255, 255, 255));
//         // Draw a grid to visualize the transformation
//         for (int i = 0; i < image_mat.rows; i += 16) cv::line(image_mat, {0, i}, {image_mat.cols, i}, {0, 0, 0}, 1);
//         for (int i = 0; i < image_mat.cols; i += 16) cv::line(image_mat, {i, 0}, {i, image_mat.rows}, {0, 0, 0}, 1);
//     }
//     torch::Tensor image = xt::utils::image::mat_to_tensor_float(image_mat);
//
//     std::cout << "--- Applying Random Affine Transformation ---" << std::endl;
//
//     // 2. Define transform: 30 deg rotation, 10% translation, 20% scale, 15 deg shear
//     xt::transforms::image::RandomAffine transformer(
//         /*degrees=*/30.0,
//         /*translate=*/std::make_pair(0.1, 0.1),
//         /*scale=*/std::make_pair(0.8, 1.2),
//         /*shear=*/std::make_pair(15.0, 15.0),
//         /*p=*/1.0, // Apply every time for demonstration
//         /*fill=*/{0.5, 0.5, 0.5} // Fill with gray
//     );
//
//     // 3. Apply the transform
//     torch::Tensor transformed_tensor = std::any_cast<torch::Tensor>(transformer.forward({image}));
//
//     // 4. Save the result
//     cv::Mat transformed_mat = xt::utils::image::tensor_to_mat_8u(transformed_tensor);
//     cv::imwrite("affine_transformed_image.png", transformed_mat);
//     std::cout << "Saved affine_transformed_image.png" << std::endl;
//
//     return 0;
// }

namespace xt::transforms::image {

    RandomAffine::RandomAffine() : RandomAffine(0.0) {}

    RandomAffine::RandomAffine(
        double degrees,
        std::optional<std::pair<double, double>> translate,
        std::optional<std::pair<double, double>> scale,
        std::optional<std::pair<double, double>> shear,
        double p,
        const std::vector<double>& fill,
        const std::string& interpolation)
        : degrees_(std::abs(degrees)),
          translate_(translate),
          scale_(scale),
          shear_(shear),
          p_(p) {

        // --- Parameter Validation ---
        if (p_ < 0.0 || p_ > 1.0) {
            throw std::invalid_argument("Probability p must be between 0.0 and 1.0.");
        }
        if (scale_.has_value() && (scale_->first <= 0 || scale_->second <= 0)) {
            throw std::invalid_argument("Scale factors must be positive.");
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
        } else if (interpolation == "bicubic") {
            interpolation_flag_ = cv::INTER_CUBIC;
        } else {
            throw std::invalid_argument("Unsupported interpolation type.");
        }

        std::random_device rd;
        gen_.seed(rd());
    }

    cv::Mat RandomAffine::get_random_transform_matrix(int width, int height) {
        // --- Generate Random Parameters ---
        std::uniform_real_distribution<> rot_dist(-degrees_, degrees_);
        double angle = rot_dist(gen_);

        double scale = 1.0;
        if (scale_.has_value()) {
            std::uniform_real_distribution<> scale_dist(scale_->first, scale_->second);
            scale = scale_dist(gen_);
        }

        double tx = 0.0, ty = 0.0;
        if (translate_.has_value()) {
            std::uniform_real_distribution<> tx_dist(-translate_->first, translate_->first);
            std::uniform_real_distribution<> ty_dist(-translate_->second, translate_->second);
            tx = tx_dist(gen_) * width;
            ty = ty_dist(gen_) * height;
        }

        double shear_x = 0.0, shear_y = 0.0;
        if (shear_.has_value()) {
            std::uniform_real_distribution<> sx_dist(-shear_->first, shear_->first);
            std::uniform_real_distribution<> sy_dist(-shear_->second, shear_->second);
            shear_x = sx_dist(gen_);
            shear_y = sy_dist(gen_);
        }

        // --- Compose the Affine Matrix ---
        // This process correctly combines transformations around the image center.

        // Center of the image
        cv::Point2f center(width / 2.0f, height / 2.0f);

        // 1. Get rotation matrix (which also includes scaling)
        cv::Mat M = cv::getRotationMatrix2D(center, angle, scale);

        // 2. Add translation
        M.at<double>(0, 2) += tx;
        M.at<double>(1, 2) += ty;

        // 3. Incorporate shear by creating a shear matrix and multiplying
        // This is equivalent to post-multiplying the current transform matrix
        // by a shear matrix, effectively shearing the rotated/scaled/translated image.
        cv::Mat shear_mat = cv::Mat::eye(2, 3, CV_64F);
        // We use tan of the angle for the shear matrix
        shear_mat.at<double>(0, 1) = std::tan(shear_x * CV_PI / 180.0);
        shear_mat.at<double>(1, 0) = std::tan(shear_y * CV_PI / 180.0);

        // To combine, we need to use 3x3 matrices
        cv::Mat M_3x3 = cv::Mat::eye(3, 3, CV_64F);
        M.copyTo(M_3x3(cv::Rect(0, 0, 3, 2)));

        cv::Mat shear_3x3 = cv::Mat::eye(3, 3, CV_64F);
        shear_mat(cv::Rect(0,0,3,2)).copyTo(shear_3x3(cv::Rect(0,0,3,2)));

        // Correctly apply shear around the center.
        cv::Mat T_neg = cv::Mat::eye(3, 3, CV_64F); // Translate to origin
        T_neg.at<double>(0,2) = -center.x;
        T_neg.at<double>(1,2) = -center.y;
        cv::Mat T_pos = cv::Mat::eye(3, 3, CV_64F); // Translate back
        T_pos.at<double>(0,2) = center.x;
        T_pos.at<double>(1,2) = center.y;

        M_3x3 = T_pos * shear_3x3 * T_neg * M_3x3;

        // Extract the final 2x3 matrix
        return M_3x3(cv::Rect(0, 0, 3, 2));
    }


    auto RandomAffine::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- Probability Check ---
        std::uniform_real_distribution<> prob_dist(0.0, 1.0);
        if (prob_dist(gen_) >= p_) {
            return tensors.begin()[0]; // Return the original tensor
        }

        // --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("RandomAffine::forward received an empty list.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to RandomAffine is not defined.");
        }

        // --- Convert to OpenCV Mat (Float for precision) ---
        cv::Mat input_mat_32f = xt::utils::image::tensor_to_mat_float(input_tensor);

        // --- Get Transformation Matrix ---
        cv::Mat affine_matrix = get_random_transform_matrix(input_mat_32f.cols, input_mat_32f.rows);

        // --- Apply the Transformation ---
        cv::Mat output_mat;
        cv::warpAffine(
            input_mat_32f,
            output_mat,
            affine_matrix,
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