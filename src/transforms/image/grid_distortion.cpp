#include <transforms/image/grid_distortion.h>

// #include "transforms/image/grid_distortion.h"
// #include <iostream>
//
// int main() {
//     // 1. Create a dummy image tensor with a grid pattern to visualize the distortion
//     torch::Tensor image = torch::zeros({3, 200, 200});
//     for (int i = 0; i < 200; i += 20) {
//         image.slice(1, i, i + 5) = 1.0; // Horizontal lines
//         image.slice(2, i, i + 5) = 1.0; // Vertical lines
//     }
//
//     // 2. Instantiate the transform
//     // A 5x5 grid with a distortion limit of 30% of the cell size.
//     xt::transforms::image::GridDistortion transformer(
//         /*num_steps=*/5,
//         /*distort_limit=*/0.3f
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
//     // The straight lines of the grid will become wavy and distorted.
//     // cv::Mat original_mat = xt::utils::image::tensor_to_mat_8u(image);
//     // cv::imwrite("original_grid_for_distortion.png", original_mat);
//     //
//     // cv::Mat distorted_mat = xt::utils::image::tensor_to_mat_8u(distorted_image);
//     // cv::imwrite("distorted_grid.png", distorted_mat);
//
//     return 0;
// }

namespace xt::transforms::image {

    GridDistortion::GridDistortion()
        : num_steps_(5), distort_limit_(0.3f), interpolation_flag_(cv::INTER_LINEAR),
          border_mode_flag_(cv::BORDER_CONSTANT), fill_value_(0.0f) {}

    GridDistortion::GridDistortion(
        int num_steps, float distort_limit, const std::string& interpolation,
        const std::string& border_mode, float fill_value
    ) : num_steps_(num_steps), distort_limit_(distort_limit), fill_value_(fill_value) {

        if (num_steps_ < 2) {
            throw std::invalid_argument("GridDistortion num_steps must be at least 2.");
        }
        if (distort_limit_ < 0.0f) {
            throw std::invalid_argument("GridDistortion distort_limit must be non-negative.");
        }

        if (interpolation == "linear") interpolation_flag_ = cv::INTER_LINEAR;
        else if (interpolation == "nearest") interpolation_flag_ = cv::INTER_NEAREST;
        else if (interpolation == "cubic") interpolation_flag_ = cv::INTER_CUBIC;
        else throw std::invalid_argument("Unsupported interpolation method for GridDistortion.");

        if (border_mode == "constant") border_mode_flag_ = cv::BORDER_CONSTANT;
        else if (border_mode == "replicate") border_mode_flag_ = cv::BORDER_REPLICATE;
        else if (border_mode == "reflect") border_mode_flag_ = cv::BORDER_REFLECT;
        else if (border_mode == "wrap") border_mode_flag_ = cv::BORDER_WRAP;
        else throw std::invalid_argument("Unsupported border mode for GridDistortion.");
    }

    auto GridDistortion::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation and Conversion to Mat ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("GridDistortion::forward received an empty list of tensors.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to GridDistortion is not defined.");
        }

        cv::Mat input_mat = xt::utils::image::tensor_to_mat_local(input_tensor);
        int height = input_mat.rows;
        int width = input_mat.cols;

        // 2. --- Define the Original and Distorted Grids ---
        float step_x = static_cast<float>(width) / num_steps_;
        float step_y = static_cast<float>(height) / num_steps_;

        std::vector<cv::Point2f> source_points;
        std::vector<cv::Point2f> dest_points;

        for (int i = 0; i <= num_steps_; ++i) {
            for (int j = 0; j <= num_steps_; ++j) {
                // Original grid point
                source_points.emplace_back(j * step_x, i * step_y);

                // Random displacement for the destination point
                float dx = cv::theRNG().uniform(-distort_limit_, distort_limit_) * step_x;
                float dy = cv::theRNG().uniform(-distort_limit_, distort_limit_) * step_y;

                // Destination grid point
                float new_x = j * step_x + dx;
                float new_y = i * step_y + dy;

                // Clamp to stay within image boundaries
                new_x = std::min((float)width, std::max(0.0f, new_x));
                new_y = std::min((float)height, std::max(0.0f, new_y));

                dest_points.emplace_back(new_x, new_y);
            }
        }

        // 3. --- Create a Dense Remapping Grid ---
        // This is the most complex part. We need to interpolate from the sparse
        // distorted grid points to create a dense map for every pixel.
        // A simple way is to use `cv::remap` with a transformation map derived
        // from interpolating the grid. A more direct approach is often
        // to use techniques like Thin Plate Splines, but a simpler interpolation works well too.

        // For this implementation, we will create a dense map by interpolating
        // between the grid points. Let's create an identity map first.
        cv::Mat map_x(input_mat.size(), CV_32F);
        cv::Mat map_y(input_mat.size(), CV_32F);

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                map_x.at<float>(i, j) = static_cast<float>(j);
                map_y.at<float>(i, j) = static_cast<float>(i);
            }
        }

        // We can create the final map by using `cv::resize` on a small map of the
        // displacement vectors, which is a very efficient way to do bilinear interpolation.
        cv::Mat displacement_x(num_steps_ + 1, num_steps_ + 1, CV_32F);
        cv::Mat displacement_y(num_steps_ + 1, num_steps_ + 1, CV_32F);

        for (int i = 0; i <= num_steps_; ++i) {
            for (int j = 0; j <= num_steps_; ++j) {
                displacement_x.at<float>(i, j) = dest_points[i * (num_steps_ + 1) + j].x - source_points[i * (num_steps_ + 1) + j].x;
                displacement_y.at<float>(i, j) = dest_points[i * (num_steps_ + 1) + j].y - source_points[i * (num_steps_ + 1) + j].y;
            }
        }

        cv::resize(displacement_x, displacement_x, input_mat.size(), 0, 0, cv::INTER_LINEAR);
        cv::resize(displacement_y, displacement_y, input_mat.size(), 0, 0, cv::INTER_LINEAR);

        map_x += displacement_x;
        map_y += displacement_y;

        // 4. --- Apply the Remapping ---
        cv::Mat transformed_mat;
        cv::remap(
            input_mat,
            transformed_mat,
            map_x,
            map_y,
            interpolation_flag_,
            border_mode_flag_,
            cv::Scalar::all(fill_value_)
        );

        // 5. --- Convert back to LibTorch Tensor ---
        torch::Tensor output_tensor = xt::utils::image::mat_to_tensor_local(transformed_mat);

        return output_tensor;
    }

} // namespace xt::transforms::image