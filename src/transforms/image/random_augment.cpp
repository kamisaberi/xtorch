#include "include/transforms/image/random_augment.h"


// --- Example Main (for testing) ---
// #include "transforms/image/random_augment.h"
// #include <iostream>
// #include <opencv2/highgui.hpp>
// #include "utils/image_conversion.h"
//
// int main() {
//     // 1. Create a sample grid image to visualize transformations clearly.
//     cv::Mat image_mat(256, 256, CV_8UC3, cv::Scalar(255, 255, 255));
//     for (int i = 0; i < image_mat.rows; i += 16) cv::line(image_mat, {0, i}, {image_mat.cols, i}, {0, 0, 0}, 1);
//     for (int i = 0; i < image_mat.cols; i += 16) cv::line(image_mat, {i, 0}, {i, image_mat.rows}, {0, 0, 0}, 1);
//     cv::putText(image_mat, "Xt", {75, 150}, cv::FONT_HERSHEY_DUPLEX, 2, {255, 0, 0}, 3);
//
//     torch::Tensor image = xt::utils::image::mat_to_tensor_float(image_mat);
//
//     std::cout << "--- Applying RandomAugment (N=3, M=15) ---" << std::endl;
//
//     // 2. Define transform: Apply 3 random ops with a strong magnitude of 15.
//     xt::transforms::image::RandomAugment augmentor(
//         /*num_ops=*/3,
//         /*magnitude=*/15,
//         /*fill=*/{0.5, 0.5, 0.5} // Fill with gray
//     );
//
//     // 3. Apply the transform
//     torch::Tensor transformed_tensor = std::any_cast<torch::Tensor>(augmentor.forward({image}));
//
//     // 4. Save the result
//     cv::Mat transformed_mat = xt::utils::image::tensor_to_mat_8u(transformed_tensor);
//     cv::imwrite("random_augment_image.png", transformed_mat);
//     std::cout << "Saved random_augment_image.png" << std::endl;
//
//     return 0;
// }


namespace xt::transforms::image {

    RandomAugment::RandomAugment() : RandomAugment(2, 9) {}

    RandomAugment::RandomAugment(
        int num_ops,
        int magnitude,
        const std::vector<double>& fill,
        const std::string& interpolation)
        : num_ops_(num_ops), magnitude_(magnitude) {

        // --- Parameter Validation ---
        if (num_ops_ < 1) {
            throw std::invalid_argument("Number of operations must be at least 1.");
        }
        if (magnitude_ < 0 || magnitude_ > 30) {
            // Common range for magnitude is 0-30.
            throw std::invalid_argument("Magnitude must be between 0 and 30.");
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

        initialize_augmentation_space();
    }

    void RandomAugment::initialize_augmentation_space() {
        // This is where we define all possible augmentations and their behaviors.
        // The level is a signed value based on magnitude. For operations without a
        // natural sign (like brightness), we'll ignore it.
        const int num_bins = 31; // Magnitude is in [0, 30]
        double level = (double)magnitude_ / (num_bins - 1);

        // A helper lambda for affine transforms
        auto affine_transformer = [this](torch::Tensor img, const cv::Mat& M) {
            cv::Mat mat = xt::utils::image::tensor_to_mat_float(img);
            cv::Mat warped_mat;
            cv::warpAffine(mat, warped_mat, M, mat.size(), interpolation_flag_, cv::BORDER_CONSTANT, fill_color_);
            return xt::utils::image::mat_to_tensor_float(warped_mat);
        };

        augmentation_space_["Identity"] = [](torch::Tensor img, double /*level*/) {
            return img;
        };
        augmentation_space_["ShearX"] = [=](torch::Tensor img, double sign) {
            double shear_factor = level * 0.3 * sign; // Max shear ~17 degrees
            cv::Mat M = (cv::Mat_<double>(2, 3) << 1, shear_factor, 0, 0, 1, 0);
            return affine_transformer(img, M);
        };
        augmentation_space_["ShearY"] = [=](torch::Tensor img, double sign) {
            double shear_factor = level * 0.3 * sign;
            cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, 0, shear_factor, 1, 0);
            return affine_transformer(img, M);
        };
        augmentation_space_["TranslateX"] = [=](torch::Tensor img, double sign) {
            double translate_factor = level * (img.size(2) / 3.0) * sign; // Max translate 1/3 image width
            cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, translate_factor, 0, 1, 0);
            return affine_transformer(img, M);
        };
        augmentation_space_["TranslateY"] = [=](torch::Tensor img, double sign) {
            double translate_factor = level * (img.size(1) / 3.0) * sign; // Max translate 1/3 image height
            cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, 0, 0, 1, translate_factor);
            return affine_transformer(img, M);
        };
        augmentation_space_["Rotate"] = [=](torch::Tensor img, double sign) {
            double angle = level * 30.0 * sign; // Max rotation 30 degrees
            cv::Mat mat = xt::utils::image::tensor_to_mat_float(img);
            cv::Point2f center(mat.cols / 2.0f, mat.rows / 2.0f);
            cv::Mat M = cv::getRotationMatrix2D(center, angle, 1.0);
            return affine_transformer(img, M);
        };
        augmentation_space_["Posterize"] = [=](torch::Tensor img, double /*sign*/) {
            // Level 0 -> keep 8 bits. Level 1 -> keep 4 bits.
            int bits_to_keep = 8 - static_cast<int>(level * 4);
            bits_to_keep = std::max(4, bits_to_keep); // Ensure we keep at least 4 bits
            cv::Mat mat_8u = xt::utils::image::tensor_to_mat_8u(img);
            uchar mask = ~((1 << (8 - bits_to_keep)) - 1);
            cv::Mat posterized_mat;
            cv::bitwise_and(mat_8u, cv::Scalar::all(mask), posterized_mat);
            return xt::utils::image::mat_to_tensor_float(posterized_mat);
        };
        augmentation_space_["Solarize"] = [=](torch::Tensor img, double /*sign*/) {
            // Level 0 -> th=255. Level 1 -> th=0. Invert pixels above threshold.
            double threshold = 1.0 - level;
            return torch::where(img < threshold, img, 1.0 - img);
        };

        auto blend_op = [](torch::Tensor img, torch::Tensor blended_img, double factor) {
            return torch::lerp(blended_img, img, factor); // img * factor + blended * (1-factor)
        };

        augmentation_space_["Contrast"] = [=](torch::Tensor img, double sign) {
            double factor = 1.0 + level * 0.9 * sign; // Range roughly [0.1, 1.9]
            cv::Mat mat_8u = xt::utils::image::tensor_to_mat_8u(img);
            cv::Mat gray_mat;
            cv::cvtColor(mat_8u, gray_mat, cv::COLOR_RGB2GRAY);
            torch::Tensor mean_img = torch::full_like(img, cv::mean(gray_mat)[0] / 255.0);
            return blend_op(img, mean_img, factor);
        };
        augmentation_space_["Color"] = [=](torch::Tensor img, double sign) { // a.k.a. Saturation
            double factor = 1.0 + level * 0.9 * sign;
            cv::Mat mat_8u = xt::utils::image::tensor_to_mat_8u(img);
            cv::Mat gray_mat;
            cv::cvtColor(mat_8u, gray_mat, cv::COLOR_RGB2GRAY);
            cv::cvtColor(gray_mat, gray_mat, cv::COLOR_GRAY2RGB); // Back to 3 channels
            torch::Tensor grayscale_tensor = xt::utils::image::mat_to_tensor_float(gray_mat);
            return blend_op(img, grayscale_tensor, factor);
        };
        augmentation_space_["Brightness"] = [=](torch::Tensor img, double sign) {
            double factor = 1.0 + level * 0.9 * sign;
            torch::Tensor black_img = torch::zeros_like(img);
            return blend_op(img, black_img, factor);
        };
        augmentation_space_["Sharpness"] = [=](torch::Tensor img, double sign) {
            double factor = 1.0 + level * 0.9 * sign;
            cv::Mat mat_32f = xt::utils::image::tensor_to_mat_float(img);
            cv::Mat blurred_mat;
            cv::GaussianBlur(mat_32f, blurred_mat, cv::Size(0, 0), 3);
            torch::Tensor blurred_tensor = xt::utils::image::mat_to_tensor_float(blurred_mat);
            return blend_op(img, blurred_tensor, factor);
        };

        // Populate the list of names for random sampling
        for(const auto& pair : augmentation_space_) {
            op_names_.push_back(pair.first);
        }
    }

    auto RandomAugment::forward(std::initializer_list<std::any> tensors) -> std::any {
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("RandomAugment::forward received an empty list.");
        }
        torch::Tensor img = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!img.defined()) {
            throw std::invalid_argument("Input tensor passed to RandomAugment is not defined.");
        }

        std::uniform_int_distribution<> op_dist(0, op_names_.size() - 1);
        std::uniform_int_distribution<> sign_dist(0, 1);

        for (int i = 0; i < num_ops_; ++i) {
            // 1. Pick a random operation
            std::string op_name = op_names_[op_dist(gen_)];

            // 2. Pick a random sign for the magnitude
            double sign = (sign_dist(gen_) == 1) ? 1.0 : -1.0;

            // 3. Apply the operation
            img = augmentation_space_[op_name](img, sign);
        }

        return img;
    }

} // namespace xt::transforms::image