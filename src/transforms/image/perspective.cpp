#include "include/transforms/image/perspective.h"

// #include "transforms/image/perspective.h"
// #include <iostream>
//
// int main() {
//     // 1. Create a dummy image tensor with a grid pattern to visualize the distortion
//     torch::Tensor image = torch::zeros({3, 200, 200});
//     for (int i = 0; i < 200; i += 20) {
//         image.slice(1, i, i + 5).index_put_({torch::indexing::Slice()}, 1.0); // Horizontal lines
//         image.slice(2, i, i + 5).index_put_({torch::indexing::Slice()}, 1.0); // Vertical lines
//     }
//
//     // 2. Instantiate the transform
//     // A distortion scale of 0.4 allows the corners to move up to 40% of the image size.
//     // Apply every time for the demo (p=1.0).
//     xt::transforms::image::Perspective transformer(
//         /*distortion_scale=*/0.4f,
//         /*p=*/1.0f
//     );
//
//     // 3. Apply the transform
//     std::any result_any = transformer.forward({image});
//     torch::Tensor distorted_image = std::any_cast<torch::Tensor>(result_any);
//
//     // 4. Check the output
//     std::cout << "Original image shape: " << image.sizes() << std::endl;
//     std::cout << "Distorted image shape: " << distorted_image.sizes() << std::endl;
//
//     // You could save the original and distorted images to see the effect.
//     // The straight grid lines will appear to converge or diverge as if
//     // viewed from an angle.
//     // cv::Mat original_mat = xt::utils::image::tensor_to_mat_8u(image);
//     // cv::imwrite("original_grid_for_perspective.png", original_mat);
//     //
//     // cv::Mat distorted_mat = xt::utils::image::tensor_to_mat_8u(distorted_image);
//     // cv::imwrite("distorted_perspective.png", distorted_mat);
//
//     return 0;
// }

namespace xt::transforms::image {

    Perspective::Perspective(float distortion_scale, float p, const std::string& interpolation, float fill_value)
        : distortion_scale_(distortion_scale), p_(p), fill_value_(fill_value) {

        if (p < 0.0f || p > 1.0f) {
            throw std::invalid_argument("Perspective probability must be between 0.0 and 1.0.");
        }
        if (distortion_scale_ < 0.0f) {
            throw std::invalid_argument("Perspective distortion_scale must be non-negative.");
        }

        if (interpolation == "linear") interpolation_flag_ = cv::INTER_LINEAR;
        else if (interpolation == "nearest") interpolation_flag_ = cv::INTER_NEAREST;
        else if (interpolation == "cubic") interpolation_flag_ = cv::INTER_CUBIC;
        else throw std::invalid_argument("Unsupported interpolation method for Perspective.");
    }

    // Default constructor calls the main constructor
    Perspective::Perspective() : Perspective(0.5f, 0.5f) {}

    auto Perspective::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- 1. Decide whether to apply the transform ---
        if (torch::rand({1}).item<float>() > p_) {
            return tensors.begin()[0];
        }

        // --- 2. Input Validation and Conversion to Mat ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("Perspective::forward received an empty list of tensors.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to Perspective is not defined.");
        }

        cv::Mat input_mat = xt::utils::image::tensor_to_mat_local(input_tensor);
        int height = input_mat.rows;
        int width = input_mat.cols;

        // --- 3. Define Source and Destination Points ---
        // Source points are the four corners of the original image
        cv::Point2f src_points[4];
        src_points[0] = cv::Point2f(0, 0);                     // Top-left
        src_points[1] = cv::Point2f(width - 1, 0);             // Top-right
        src_points[2] = cv::Point2f(0, height - 1);            // Bottom-left
        src_points[3] = cv::Point2f(width - 1, height - 1);  // Bottom-right

        // Destination points are randomly perturbed versions of the source points
        cv::Point2f dst_points[4];
        float max_dx = width * distortion_scale_ / 2.0f;
        float max_dy = height * distortion_scale_ / 2.0f;

        for (int i = 0; i < 4; ++i) {
            float dx = cv::theRNG().uniform(-max_dx, max_dx);
            float dy = cv::theRNG().uniform(-max_dy, max_dy);
            dst_points[i] = cv::Point2f(src_points[i].x + dx, src_points[i].y + dy);
        }

        // --- 4. Get the Transformation Matrix and Apply It ---
        cv::Mat M = cv::getPerspectiveTransform(src_points, dst_points);

        cv::Mat transformed_mat;

        cv::Scalar cv_fill_value;
        if (input_mat.channels() == 3) {
            cv_fill_value = cv::Scalar(fill_value_, fill_value_, fill_value_);
        } else {
            cv_fill_value = cv::Scalar(fill_value_);
        }

        cv::warpPerspective(
            input_mat,
            transformed_mat,
            M,
            input_mat.size(),   // Output size is the same as input
            interpolation_flag_,
            cv::BORDER_CONSTANT,
            cv_fill_value
        );

        // --- 5. Convert back to LibTorch Tensor ---
        torch::Tensor output_tensor = xt::utils::image::mat_to_tensor_local(transformed_mat);

        return output_tensor;
    }

} // namespace xt::transforms::image