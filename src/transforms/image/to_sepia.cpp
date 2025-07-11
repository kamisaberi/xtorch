#include "include/transforms/image/to_sepia.h"


// #include "transforms/image/to_sepia.h"
// #include <iostream>
//
// int main() {
//     // 1. Create a dummy color image tensor
//     torch::Tensor image = torch::rand({3, 200, 200});
//
//     std::cout << "Original image shape: " << image.sizes() << std::endl;
//
//     // 2. Instantiate the transform
//     xt::transforms::image::ToSepia sepia_toner;
//
//     // 3. Apply the transform
//     std::any result_any = sepia_toner.forward({image});
//     torch::Tensor sepia_image = std::any_cast<torch::Tensor>(result_any);
//
//     // 4. Check the output
//     std::cout << "Sepia image shape: " << sepia_image.sizes() << std::endl;
//     // The shape should be unchanged.
//
//     // You could save the output image to see the warm, brownish toning effect.
//     // cv::Mat output_mat = xt::utils::image::tensor_to_mat_8u(sepia_image);
//     // cv::imwrite("sepia_image.png", output_mat);
//
//     // We can also check that the channels are now highly correlated,
//     // which is characteristic of a monochrome-like effect.
//     auto r_channel = sepia_image[0];
//     auto g_channel = sepia_image[1];
//
//     // The mean values should follow the ratios in the sepia matrix
//     std::cout << "Mean of Red channel (sepia): " << r_channel.mean().item<float>() << std::endl;
//     std::cout << "Mean of Green channel (sepia): " << g_channel.mean().item<float>() << std::endl;
//
//     return 0;
// }

namespace xt::transforms::image {

    ToSepia::ToSepia() {
        // Define the standard Sepia transformation matrix
        sepia_matrix_ = torch::tensor({
            {0.393, 0.769, 0.189},
            {0.349, 0.686, 0.168},
            {0.272, 0.534, 0.131}
        }, torch::kFloat32);
    }

    auto ToSepia::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("ToSepia::forward received an empty list of tensors.");
        }
        torch::Tensor image = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!image.defined()) {
            throw std::invalid_argument("Input tensor passed to ToSepia is not defined.");
        }
        if (image.dim() != 3 || image.size(0) != 3) {
            throw std::invalid_argument("ToSepia expects a 3-channel (RGB) image tensor.");
        }

        // Move the sepia matrix to the same device as the input image
        sepia_matrix_ = sepia_matrix_.to(image.device());

        // 2. --- Reshape and Apply Transformation ---
        const int64_t C = image.size(0);
        const int64_t H = image.size(1);
        const int64_t W = image.size(2);

        // To perform the matrix multiplication, we need to reshape the image tensor.
        // Flatten the spatial dimensions (H, W) into a single dimension.
        // [C, H, W] -> [C, H*W]
        torch::Tensor flat_image = image.view({C, H * W});

        // Now we can perform the matrix multiplication:
        // [3, 3] @ [3, H*W] -> [3, H*W]
        torch::Tensor transformed_image = sepia_matrix_.matmul(flat_image);

        // Reshape the result back to the original image shape [C, H, W]
        transformed_image = transformed_image.view({C, H, W});

        // 3. --- Clamp to Valid Range ---
        // It's crucial to clamp the result as the transformation can produce values > 1.0
        torch::Tensor sepia_image = torch::clamp(transformed_image, 0.0, 1.0);

        return sepia_image;
    }

} // namespace xt::transforms::image