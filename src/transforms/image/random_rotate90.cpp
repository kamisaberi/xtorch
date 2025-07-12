#include "include/transforms/image/random_rotate90.h"



// --- Example Main (for testing) ---
// #include "transforms/image/random_rotate90.h"
// #include <iostream>
// #include <opencv2/highgui.hpp>
// #include "utils/image_conversion.h"
//
// int main() {
//     // 1. Create a non-square image with an 'F' to see the rotation clearly.
//     cv::Mat image_mat(200, 300, CV_8UC3, cv::Scalar(255, 255, 255));
//     // Draw a large 'F'
//     cv::rectangle(image_mat, {50, 50}, {80, 150}, {0,0,255}, -1);
//     cv::rectangle(image_mat, {80, 50}, {180, 80}, {0,0,255}, -1);
//     cv::rectangle(image_mat, {80, 100}, {150, 120}, {0,0,255}, -1);
//
//     cv::imwrite("rotate90_before.png", image_mat);
//     std::cout << "Saved rotate90_before.png (Shape: " << image_mat.size() << ")" << std::endl;
//
//     torch::Tensor image = xt::utils::image::mat_to_tensor_float(image_mat);
//
//     std::cout << "--- Applying RandomRotate90 ---" << std::endl;
//
//     // 2. Define the transform, with p=1.0 to guarantee it runs.
//     xt::transforms::image::RandomRotate90 rotator(1.0);
//
//     // 3. Apply the transform
//     torch::Tensor transformed_tensor = std::any_cast<torch::Tensor>(rotator.forward({image}));
//
//     // 4. Save the result
//     cv::Mat transformed_mat = xt::utils::image::tensor_to_mat_8u(transformed_tensor);
//     cv::imwrite("rotate90_after.png", transformed_mat);
//     std::cout << "Saved rotate90_after.png (Shape: " << transformed_mat.size() << ")" << std::endl;
//
//     return 0;
// }


namespace xt::transforms::image {

    RandomRotate90::RandomRotate90() : RandomRotate90(0.5) {}

    RandomRotate90::RandomRotate90(double p) : p_(p) {
        if (p_ < 0.0 || p_ > 1.0) {
            throw std::invalid_argument("Probability p must be between 0.0 and 1.0.");
        }
        std::random_device rd;
        gen_.seed(rd());
    }

    auto RandomRotate90::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- Probability Check ---
        std::uniform_real_distribution<> prob_dist(0.0, 1.0);
        if (prob_dist(gen_) >= p_) {
            return tensors.begin()[0]; // Return the original tensor
        }

        // --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("RandomRotate90::forward received an empty list.");
        }
        torch::Tensor img = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!img.defined()) {
            throw std::invalid_argument("Input tensor passed to RandomRotate90 is not defined.");
        }
        if (img.dim() != 3) {
            throw std::invalid_argument("RandomRotate90 expects a 3D tensor (C, H, W).");
        }

        // --- Select a random number of 90-degree rotations ---
        // k=0: 0 degrees, k=1: 90, k=2: 180, k=3: 270
        std::uniform_int_distribution<int> k_dist(0, 3);
        int k = k_dist(gen_);

        if (k == 0) {
            // No rotation needed
            return img;
        }

        // --- Apply Rotation ---
        // For a (C, H, W) tensor, the rotation is in the H-W plane, so we specify
        // dims {1, 2}.
        return torch::rot90(img, k, {1, 2});
    }

} // namespace xt::transforms::image