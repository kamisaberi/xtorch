#include <transforms/image/random_shadow.h>


// --- Example Main (for testing) ---
// #include "transforms/image/random_shadow.h"
// #include <iostream>
// #include <opencv2/highgui.hpp>
// #include "utils/image_conversion.h"
//
// int main() {
//     // 1. Create a sample image.
//     cv::Mat image_mat(300, 400, CV_8UC3, cv::Scalar(200, 220, 240)); // A light blue sky
//     // Draw a "sun"
//     cv::circle(image_mat, {350, 50}, 30, {255, 255, 0}, -1);
//     cv::imwrite("shadow_before.png", image_mat);
//     std::cout << "Saved shadow_before.png" << std::endl;
//
//     torch::Tensor image = xt::utils::image::mat_to_tensor_float(image_mat);
//
//     std::cout << "--- Applying RandomShadow ---" << std::endl;
//
//     // 2. Define transform. The shadow will reduce brightness to between 20% and 60%.
//     xt::transforms::image::RandomShadow shader({0.2, 0.6}, 1.0);
//
//     // 3. Apply the transform
//     torch::Tensor shadowed_tensor = std::any_cast<torch::Tensor>(shader.forward({image}));
//
//     // 4. Save the result
//     cv::Mat shadowed_mat = xt::utils::image::tensor_to_mat_8u(shadowed_tensor);
//     cv::imwrite("shadow_after.png", shadowed_mat);
//     std::cout << "Saved shadow_after.png" << std::endl;
//
//     return 0;
// }


namespace xt::transforms::image {

    RandomShadow::RandomShadow() : RandomShadow({0.5, 1.0}, 0.5) {}

    RandomShadow::RandomShadow(std::pair<double, double> shadow_range, double p)
        : shadow_range_(shadow_range), p_(p) {

        // --- Parameter Validation ---
        if (shadow_range_.first < 0.0 || shadow_range_.second > 1.0 || shadow_range_.first > shadow_range_.second) {
            throw std::invalid_argument("Shadow range must be valid and in [0, 1].");
        }
        if (p_ < 0.0 || p_ > 1.0) {
            throw std::invalid_argument("Probability p must be between 0.0 and 1.0.");
        }
        std::random_device rd;
        gen_.seed(rd());
    }

    auto RandomShadow::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- Probability Check ---
        std::uniform_real_distribution<> prob_dist(0.0, 1.0);
        if (prob_dist(gen_) >= p_) {
            return tensors.begin()[0]; // Return the original tensor
        }

        // --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("RandomShadow::forward received an empty list.");
        }
        torch::Tensor img = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!img.defined()) {
            throw std::invalid_argument("Input tensor passed to RandomShadow is not defined.");
        }

        auto h = img.size(1);
        auto w = img.size(2);

        // --- Generate Shadow Polygon ---
        // We define a quadrilateral to represent the shadow.
        // Two vertices will be on one side of the image, two on the opposite.
        std::vector<cv::Point> polygon(4);
        std::uniform_int_distribution<> side_dist(0, 3); // 0: top, 1: bottom, 2: left, 3: right
        int side1 = side_dist(gen_);
        int side2 = (side1 % 2 == 0) ? 1 - side1 : 5 - side1; // Get opposite side

        std::uniform_real_distribution<> dist(0.0, 1.0);

        // Vertices for the first side
        if (side1 == 0) { // Top
            polygon[0] = cv::Point(w * dist(gen_), 0);
            polygon[1] = cv::Point(w * dist(gen_), 0);
        } else if (side1 == 1) { // Bottom
            polygon[0] = cv::Point(w * dist(gen_), h - 1);
            polygon[1] = cv::Point(w * dist(gen_), h - 1);
        } else if (side1 == 2) { // Left
            polygon[0] = cv::Point(0, h * dist(gen_));
            polygon[1] = cv::Point(0, h * dist(gen_));
        } else { // Right
            polygon[0] = cv::Point(w - 1, h * dist(gen_));
            polygon[1] = cv::Point(w - 1, h * dist(gen_));
        }

        // Vertices for the second (opposite) side
        if (side2 == 0) { // Top
            polygon[2] = cv::Point(w * dist(gen_), 0);
            polygon[3] = cv::Point(w * dist(gen_), 0);
        } else if (side2 == 1) { // Bottom
            polygon[2] = cv::Point(w * dist(gen_), h - 1);
            polygon[3] = cv::Point(w * dist(gen_), h - 1);
        } else if (side2 == 2) { // Left
            polygon[2] = cv::Point(0, h * dist(gen_));
            polygon[3] = cv::Point(0, h * dist(gen_));
        } else { // Right
            polygon[3] = cv::Point(w - 1, h * dist(gen_));
            polygon[2] = cv::Point(w - 1, h * dist(gen_));
        }

        // --- Create Shadow Mask ---
        // Start with a white mask (1.0) and draw a semi-transparent polygon on it.
        cv::Mat mask_mat(h, w, CV_32FC1, cv::Scalar(1.0));
        std::uniform_real_distribution<> shadow_dist(shadow_range_.first, shadow_range_.second);
        double shadow_factor = shadow_dist(gen_);

        // Draw the filled polygon with the shadow factor value.
        cv::fillConvexPoly(mask_mat, polygon, cv::Scalar(shadow_factor));

        // Optionally add some blur to make the shadow edges softer
        cv::GaussianBlur(mask_mat, mask_mat, cv::Size(0, 0), 25);

        // Convert the OpenCV mask to a LibTorch tensor
        torch::Tensor mask_tensor = xt::utils::image::mat_to_tensor_float(mask_mat);

        // --- Apply the Shadow ---
        // Multiply the original image by the shadow mask.
        // The mask is [1, H, W], it will be broadcast across the 3 color channels.
        torch::Tensor output_tensor = img * mask_tensor;

        return output_tensor;
    }

} // namespace xt::transforms::image