#include "include/transforms/image/optical_distortion.h"



namespace xt::transforms::image {

    OpticalDistortion::OpticalDistortion(
        float distort_limit, float shift_limit, const std::string& interpolation,
        const std::string& border_mode, float fill_value
    ) : distort_limit_(distort_limit), shift_limit_(shift_limit), fill_value_(fill_value) {

        if (interpolation == "linear") interpolation_flag_ = cv::INTER_LINEAR;
        else if (interpolation == "nearest") interpolation_flag_ = cv::INTER_NEAREST;
        else if (interpolation == "cubic") interpolation_flag_ = cv::INTER_CUBIC;
        else throw std::invalid_argument("Unsupported interpolation method for OpticalDistortion.");

        if (border_mode == "constant") border_mode_flag_ = cv::BORDER_CONSTANT;
        else if (border_mode == "replicate") border_mode_flag_ = cv::BORDER_REPLICATE;
        else if (border_mode == "reflect") border_mode_flag_ = cv::BORDER_REFLECT;
        else if (border_mode == "wrap") border_mode_flag_ = cv::BORDER_WRAP;
        else throw std::invalid_argument("Unsupported border mode for OpticalDistortion.");
    }

    // Default constructor calls the main constructor with default values
    OpticalDistortion::OpticalDistortion() : OpticalDistortion(0.5f, 0.5f) {}

    auto OpticalDistortion::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation and Conversion to Mat ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("OpticalDistortion::forward received an empty list of tensors.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to OpticalDistortion is not defined.");
        }

        cv::Mat input_mat = xt::utils::image::tensor_to_mat_local(input_tensor);
        int height = input_mat.rows;
        int width = input_mat.cols;

        // 2. --- Create a Virtual Camera and Distortion Model ---
        // Create an identity camera matrix, assuming the camera is looking straight ahead.
        cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) <<
            width, 0, width / 2.0,
            0, width, height / 2.0, // Use width for focal length for square pixels
            0, 0, 1);

        // Generate random distortion coefficients based on the user-provided limits.
        double k1 = cv::theRNG().uniform(-distort_limit_, distort_limit_);
        double k2 = cv::theRNG().uniform(-distort_limit_, distort_limit_);
        double k3 = cv::theRNG().uniform(-distort_limit_, distort_limit_);
        double p1 = cv::theRNG().uniform(-shift_limit_, shift_limit_);
        double p2 = cv::theRNG().uniform(-shift_limit_, shift_limit_);

        cv::Mat dist_coeffs = (cv::Mat_<double>(5, 1) << k1, k2, p1, p2, k3);

        // 3. --- Apply the Distortion ---
        // We use `cv::remap` for this, as it's more flexible than `undistort`.
        // First, we need to compute the mapping from distorted to undistorted points.
        cv::Mat map1, map2;
        cv::initUndistortRectifyMap(
            camera_matrix,
            dist_coeffs,
            cv::Mat(), // No rectification
            camera_matrix, // New camera matrix is the same as the old one
            cv::Size(width, height),
            CV_32FC1,
            map1, map2
        );

        cv::Mat transformed_mat;
        cv::remap(
            input_mat,
            transformed_mat,
            map1,
            map2,
            interpolation_flag_,
            border_mode_flag_,
            cv::Scalar::all(fill_value_)
        );

        // 4. --- Convert back to LibTorch Tensor ---
        torch::Tensor output_tensor = xt::utils::image::mat_to_tensor_local(transformed_mat);

        return output_tensor;
    }

} // namespace xt::transforms::image