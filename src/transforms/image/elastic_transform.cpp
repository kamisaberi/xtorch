#include "include/transforms/image/elastic_transform.h"



// #include "transforms/image/elastic_transform.h"
// #include <iostream>
//
// int main() {
//     // 1. Create a dummy image tensor with a grid pattern to visualize the distortion
//     torch::Tensor image = torch::zeros({3, 200, 200});
//     for (int i = 0; i < 200; i += 20) {
//         image.slice(1, i, i + 10) = 1.0; // Horizontal lines
//         image.slice(2, i, i + 10) = 1.0; // Vertical lines
//     }
//
//     // 2. Instantiate the transform
//     // A high alpha and low sigma will create intense, wavy distortions.
//     xt::transforms::image::ElasticTransform transformer(
//         /*alpha=*/60.0,
//         /*sigma=*/4.0
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
//     // cv::Mat original_mat = xt::utils::image::tensor_to_mat_local(image);
//     // cv::imwrite("original_grid.png", original_mat * 255);
//     // cv::Mat distorted_mat = xt::utils::image::tensor_to_mat_local(distorted_image);
//     // cv::imwrite("distorted_grid.png", distorted_mat * 255);
//
//     return 0;
// }



namespace xt::transforms::image {

    ElasticTransform::ElasticTransform()
        : alpha_(34.0), sigma_(4.0), interpolation_flag_(cv::INTER_LINEAR),
          border_mode_flag_(cv::BORDER_CONSTANT), fill_value_(0.0f) {}

    ElasticTransform::ElasticTransform(
        double alpha, double sigma, const std::string& interpolation,
        const std::string& border_mode, float fill_value
    ) : alpha_(alpha), sigma_(sigma), fill_value_(fill_value) {

        if (interpolation == "linear") interpolation_flag_ = cv::INTER_LINEAR;
        else if (interpolation == "nearest") interpolation_flag_ = cv::INTER_NEAREST;
        else if (interpolation == "cubic") interpolation_flag_ = cv::INTER_CUBIC;
        else throw std::invalid_argument("Unsupported interpolation method for ElasticTransform.");

        if (border_mode == "constant") border_mode_flag_ = cv::BORDER_CONSTANT;
        else if (border_mode == "replicate") border_mode_flag_ = cv::BORDER_REPLICATE;
        else if (border_mode == "reflect") border_mode_flag_ = cv::BORDER_REFLECT;
        else if (border_mode == "wrap") border_mode_flag_ = cv::BORDER_WRAP;
        else throw std::invalid_argument("Unsupported border mode for ElasticTransform.");
    }

    auto ElasticTransform::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation and Conversion to Mat ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("ElasticTransform::forward received an empty list of tensors.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to ElasticTransform is not defined.");
        }

        cv::Mat input_mat = xt::utils::image::tensor_to_mat_local(input_tensor);
        cv::Size size = input_mat.size();

        // 2. --- Generate Random Displacement Fields ---
        // Create two random fields (for x and y) with values from -1 to 1.
        cv::Mat dx(size, CV_32F);
        cv::randu(dx, cv::Scalar::all(-1), cv::Scalar::all(1));
        cv::Mat dy(size, CV_32F);
        cv::randu(dy, cv::Scalar::all(-1), cv::Scalar::all(1));

        // 3. --- Smooth the Displacement Fields ---
        // Use a Gaussian blur. Sigma controls the "stiffness" of the elastic material.
        cv::GaussianBlur(dx, dx, cv::Size(0, 0), sigma_);
        cv::GaussianBlur(dy, dy, cv::Size(0, 0), sigma_);

        // 4. --- Create the Remapping Grid ---
        // Create an identity mapping grid.
        cv::Mat map_x(size, CV_32F), map_y(size, CV_32F);
        for (int i = 0; i < size.height; ++i) {
            for (int j = 0; j < size.width; ++j) {
                map_x.at<float>(i, j) = static_cast<float>(j);
                map_y.at<float>(i, j) = static_cast<float>(i);
            }
        }

        // Add the scaled, blurred displacements to the identity grid.
        // Alpha controls the intensity of the deformation.
        map_x += dx * alpha_;
        map_y += dy * alpha_;

        // 5. --- Apply the Remapping ---
        cv::Mat transformed_mat;
        cv::remap(
            input_mat,
            transformed_mat,
            map_x,
            map_y,
            interpolation_flag_,
            border_mode_flag_,
            cv::Scalar::all(fill_value_) // Fill value for constant border mode
        );

        // 6. --- Convert back to LibTorch Tensor ---
        torch::Tensor output_tensor = xt::utils::image::mat_to_tensor_local(transformed_mat);

        return output_tensor;
    }

} // namespace xt::transforms::image