#include <transforms/image/piecewise_affine.h>

//
// #include "transforms/image/piecewise_affine.h"
// #include <iostream>
//
// int main() {
//     // 1. Create a dummy image tensor with a grid pattern to visualize the distortion
//     torch::Tensor image = torch::zeros({3, 224, 224});
//     for (int i = 0; i < 224; i += 20) {
//         image.slice(1, i, i + 5).index_put_({torch::indexing::Slice()}, 1.0); // Horizontal lines
//         image.slice(2, i, i + 5).index_put_({torch::indexing::Slice()}, 1.0); // Vertical lines
//     }
//
//     // 2. Instantiate the transform
//     // A 5x5 grid of points with a moderate distortion scale of 5% of the image size.
//     xt::transforms::image::PiecewiseAffine transformer(
//         /*scale=*/0.05f,
//         /*nb_rows=*/5,
//         /*nb_cols=*/5,
//         /*p=*/1.0f // Apply every time for demo
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
//     // The straight lines of the grid will become wavy and distorted in a localized manner.
//     // cv::Mat original_mat = xt::utils::image::tensor_to_mat_8u(image);
//     // cv::imwrite("original_grid_for_piecewise.png", original_mat);
//     //
//     // cv::Mat distorted_mat = xt::utils::image::tensor_to_mat_8u(distorted_image);
//     // cv::imwrite("distorted_piecewise.png", distorted_mat);
//
//     return 0;
// }

namespace xt::transforms::image {

    PiecewiseAffine::PiecewiseAffine(float scale, int nb_rows, int nb_cols, float p)
        : scale_(scale), nb_rows_(nb_rows), nb_cols_(nb_cols), p_(p) {

        if (p_ < 0.0f || p_ > 1.0f) {
            throw std::invalid_argument("PiecewiseAffine probability must be between 0.0 and 1.0.");
        }
        if (scale_ < 0.0f) {
            throw std::invalid_argument("PiecewiseAffine scale must be non-negative.");
        }
        if (nb_rows_ < 2 || nb_cols_ < 2) {
            throw std::invalid_argument("PiecewiseAffine must have at least 2 rows and 2 columns.");
        }
    }

    // Default constructor
    PiecewiseAffine::PiecewiseAffine() : PiecewiseAffine(0.05f, 4, 4, 0.5f) {}

    auto PiecewiseAffine::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- 1. Decide whether to apply the transform ---
        if (torch::rand({1}).item<float>() > p_) {
            return tensors.begin()[0];
        }

        // --- 2. Input Validation and Conversion to Mat ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("PiecewiseAffine::forward received an empty list of tensors.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to PiecewiseAffine is not defined.");
        }

        cv::Mat input_mat = xt::utils::image::tensor_to_mat_local(input_tensor);
        int height = input_mat.rows;
        int width = input_mat.cols;

        // --- 3. Create Source and Destination Grid Points ---
        std::vector<cv::Point2f> src_points;
        for (int i = 0; i < nb_rows_; ++i) {
            for (int j = 0; j < nb_cols_; ++j) {
                float x = static_cast<float>(j) * (width - 1) / (nb_cols_ - 1);
                float y = static_cast<float>(i) * (height - 1) / (nb_rows_ - 1);
                src_points.emplace_back(x, y);
            }
        }

        // Create randomly perturbed destination points
        std::vector<cv::Point2f> dst_points = src_points;
        float max_offset_x = width * scale_;
        float max_offset_y = height * scale_;
        for (size_t i = 0; i < dst_points.size(); ++i) {
            // Don't move the corner points to avoid weird border effects
            if (dst_points[i].x != 0 && dst_points[i].x != width - 1 &&
                dst_points[i].y != 0 && dst_points[i].y != height - 1) {
                dst_points[i].x += cv::theRNG().uniform(-max_offset_x, max_offset_x);
                dst_points[i].y += cv::theRNG().uniform(-max_offset_y, max_offset_y);
            }
        }

        // --- 4. Perform Delaunay Triangulation ---
        cv::Rect rect(0, 0, width, height);
        cv::Subdiv2D subdiv(rect);
        subdiv.insert(src_points);

        std::vector<cv::Vec6f> triangle_list;
        subdiv.getTriangleList(triangle_list);

        // --- 5. Warp each triangle individually ---
        cv::Mat transformed_mat = cv::Mat::zeros(input_mat.size(), input_mat.type());

        for (const auto& t : triangle_list) {
            // Find the corresponding source and destination triangles
            cv::Point2f src_tri[3], dst_tri[3];
            src_tri[0] = cv::Point2f(t[0], t[1]);
            src_tri[1] = cv::Point2f(t[2], t[3]);
            src_tri[2] = cv::Point2f(t[4], t[5]);

            bool found = true;
            for (int i=0; i < 3; ++i) {
                bool point_found = false;
                for (size_t j = 0; j < src_points.size(); ++j) {
                    if (src_tri[i] == src_points[j]) {
                        dst_tri[i] = dst_points[j];
                        point_found = true;
                        break;
                    }
                }
                if (!point_found) { found = false; break; }
            }
            if (!found) continue;

            // --- Warp the triangle ---
            cv::Mat M = cv::getAffineTransform(src_tri, dst_tri);

            // Create a mask for the destination triangle
            cv::Mat mask = cv::Mat::zeros(input_mat.size(), CV_8U);
            cv::Point dst_tri_int[3];
            for(int i=0; i<3; ++i) dst_tri_int[i] = dst_tri[i];
            cv::fillConvexPoly(mask, dst_tri_int, 3, cv::Scalar(255));

            // Warp the source triangle region
            cv::Mat warped_tri;
            cv::warpAffine(input_mat, warped_tri, M, input_mat.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT_101);

            // Copy the warped triangle to the final image using the mask
            warped_tri.copyTo(transformed_mat, mask);
        }

        // --- 6. Convert back to LibTorch Tensor ---
        torch::Tensor output_tensor = xt::utils::image::mat_to_tensor_local(transformed_mat);

        return output_tensor;
    }

} // namespace xt::transforms::image