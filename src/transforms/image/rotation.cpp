#include "../../../include/transforms/image/rotation.h"

namespace xt::transforms::image {

    Rotation::Rotation(double angle_deg) : angle(angle_deg) {
    }

    torch::Tensor Rotation::operator()(const torch::Tensor &input_tensor) {
        // Convert torch::Tensor to OpenCV Mat (assuming CHW format and float32 in [0,1])
        auto img_tensor = input_tensor.detach().cpu().clone();
        img_tensor = img_tensor.permute({1, 2, 0}); // Convert CHW -> HWC
        img_tensor = img_tensor.mul(255).clamp(0, 255).to(torch::kU8);

        cv::Mat img(img_tensor.size(0), img_tensor.size(1), CV_8UC3);
        std::memcpy((void *) img.data, img_tensor.data_ptr(), sizeof(uint8_t) * img_tensor.numel());

        // Compute center of the image and get rotation matrix
        cv::Point2f center(img.cols / 2.0f, img.rows / 2.0f);
        cv::Mat rot_matrix = cv::getRotationMatrix2D(center, angle, 1.0);

        // Rotate the image
        cv::Mat rotated_img;
        cv::warpAffine(img, rotated_img, rot_matrix, img.size(), cv::INTER_LINEAR);

        // Convert back to Tensor
        torch::Tensor rotated_tensor = torch::from_blob(
            rotated_img.data,
            {rotated_img.rows, rotated_img.cols, 3},
            torch::kUInt8).clone();

        rotated_tensor = rotated_tensor.permute({2, 0, 1}); // HWC -> CHW
        rotated_tensor = rotated_tensor.to(torch::kFloat32).div(255); // Normalize to [0,1]

        return rotated_tensor;
    }




        // Rotation::Rotation(float angle) : angle(angle) {}
    //
    // // Operator: Rotate the input tensor by the specified angle
    //
    //
    // torch::Tensor Rotation::operator()(torch::Tensor input) {
    //     int64_t input_dims = input.dim();
    //     if (input_dims < 3 || input_dims > 4) {
    //         throw std::runtime_error("Input tensor must be 3D ([C, H, W]) or 4D ([N, C, H, W]).");
    //     }
    //
    //     // Get spatial dimensions
    //     int64_t h = input.size(input_dims - 2);
    //     int64_t w = input.size(input_dims - 1);
    //
    //     // Convert angle to radians
    //     float rad = angle * M_PI / 180.0;
    //     float cos_val = std::cos(rad);
    //     float sin_val = std::sin(rad);
    //
    //     // Create 2x3 affine matrix: [cos, -sin, 0; sin, cos, 0]
    //     torch::Tensor theta = torch::tensor({
    //         {cos_val, -sin_val, 0.0},
    //         {sin_val,  cos_val, 0.0}
    //     }, input.options()).reshape({1, 2, 3});
    //
    //     // Repeat for batch dimension if 4D
    //     if (input_dims == 4) {
    //         theta = theta.repeat({input.size(0), 1, 1});
    //     }
    //
    //     // Generate grid for sampling
    //     torch::Tensor grid = torch::nn::functional::affine_grid(
    //         theta,
    //         input_dims == 4 ? torch::IntArrayRef({input.size(0), input.size(1), h, w})
    //                         : torch::IntArrayRef({1, input.size(0), h, w}),
    //         /*align_corners=*/false
    //     );
    //
    //     // Add batch dimension if 3D
    //     bool is_3d = (input_dims == 3);
    //     if (is_3d) {
    //         input = input.unsqueeze(0);
    //     }
    //
    //     // Sample the input with the rotated grid using correct enum types
    //     torch::Tensor output = torch::nn::functional::grid_sample(
    //         input,
    //         grid,
    //         torch::nn::functional::GridSampleFuncOptions()
    //             .interpolation_mode(torch::enumtype::kBilinear)
    //             .padding_mode(torch::enumtype::kZeros)
    //             .align_corners(false)
    //     );
    //
    //     // Remove batch dimension if added
    //     if (is_3d) {
    //         output = output.squeeze(0);
    //     }
    //
    //     return output;
    // }








}