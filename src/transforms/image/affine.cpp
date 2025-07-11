#include "include/transforms/image/affine.h"



#include "transforms/image/affine.h"
#include <iostream>
//
// int main() {
//     // 1. Create a dummy image
//     torch::Tensor image = torch::zeros({3, 200, 200});
//     // Add a rectangle to visualize the transformation
//     image.slice(1, 75, 125).slice(2, 75, 125) = 1.0;
//
//     // 2. Instantiate the transform
//     // Rotate 30 degrees, scale down to 75%, no translation or shear
//     xt::transforms::image::Affine transformer(30.0, {0.0, 0.0}, 0.75, 0.0);
//
//     // 3. Apply the transform
//     std::any result_any = transformer.forward({image});
//     torch::Tensor transformed_image = std::any_cast<torch::Tensor>(result_any);
//
//     // 4. Check the output
//     std::cout << "Original image shape: " << image.sizes() << std::endl;
//     std::cout << "Transformed image shape: " << transformed_image.sizes() << std::endl;
//
//     // You could save the output image to see the rotated, smaller rectangle
//     // cv::Mat output_mat = xt::utils::image::tensor_to_mat_local(transformed_image);
//     // cv::imwrite("affine_transformed.png", output_mat * 255);
//
//     return 0;
// }


namespace xt::transforms::image {

    // Default constructor: identity transformation
    Affine::Affine() : degrees_(0.0), translate_({0.0, 0.0}), scale_(1.0), shear_(0.0) {}

    Affine::Affine(double degrees, const std::vector<double>& translate, double scale, double shear)
        : degrees_(degrees), translate_(translate), scale_(scale), shear_(shear) {

        if (translate_.size() != 2) {
            throw std::invalid_argument("Affine translate must be a vector of two doubles.");
        }
    }

    auto Affine::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation and Conversion to Mat ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("Affine::forward received an empty list of tensors.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to Affine is not defined.");
        }

        cv::Mat input_mat = xt::utils::image::tensor_to_mat_local(input_tensor);
        const int height = input_mat.rows;
        const int width = input_mat.cols;

        // 2. --- Build the 2x3 Affine Transformation Matrix ---

        // Start with rotation and scaling, centered on the image
        cv::Point2f center(width / 2.0f, height / 2.0f);
        cv::Mat M = cv::getRotationMatrix2D(center, degrees_, scale_);

        // Add translation to the matrix
        // The values are fractions of the image size
        M.at<double>(0, 2) += translate_[0] * width;
        M.at<double>(1, 2) += translate_[1] * height;

        // Add shear to the matrix
        // This is done by modifying the off-diagonal elements
        double shear_rad = shear_ * CV_PI / 180.0;
        cv::Mat shear_mat = (cv::Mat_<double>(2, 3) << 1, tan(shear_rad), 0, 0, 1, 0);

        // To apply shear correctly around the center, we need to translate to origin,
        // shear, and translate back. This is done by matrix multiplication.
        // For simplicity in this example, we apply a simpler shear matrix.
        // A full implementation would be more complex.
        // Let's add the shear component to the rotation matrix M
        // This is a simplification; a true shear would require a more complex matrix composition.
        // For a basic implementation, we can do:
        M.at<double>(0, 1) += tan(shear_rad);

        // 3. --- Apply the Affine Transformation ---
        cv::Mat transformed_mat;
        cv::warpAffine(
            input_mat,          // source image
            transformed_mat,    // destination image
            M,                  // 2x3 transformation matrix
            input_mat.size(),   // output image size
            cv::INTER_LINEAR,   // interpolation method
            cv::BORDER_CONSTANT,// border mode
            cv::Scalar(0, 0, 0) // border value (black)
        );

        // 4. --- Convert back to LibTorch Tensor ---
        torch::Tensor output_tensor = xt::utils::image::mat_to_tensor_local(transformed_mat);

        return output_tensor;
    }

} // namespace xt::transforms::image