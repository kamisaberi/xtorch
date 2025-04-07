#include "../../include/definitions/transforms.h"

namespace xt::data::transforms {
    std::function<torch::Tensor(torch::Tensor input)> create_resize_transform(std::vector<int64_t> size) {
        auto resize_fn = [size](torch::Tensor img) -> torch::Tensor {
            img = img.unsqueeze(0); // Add batch dimension
            img = torch::nn::functional::interpolate(
                img,
                torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>({size[0], size[1]}))
                .mode(torch::kBilinear)
                .align_corners(false)
            );
            return img.squeeze(0); // Remove batch dimension
        };
        return resize_fn;
    }


    torch::Tensor resize_tensor(const torch::Tensor &tensor, const std::vector<int64_t> &size) {
        return torch::nn::functional::interpolate(
            tensor.unsqueeze(0),
            torch::nn::functional::InterpolateFuncOptions().size(size).mode(torch::kBilinear).align_corners(false)
        ).squeeze(0);
    }

    torch::data::transforms::Lambda<torch::data::Example<> > resize(std::vector<int64_t> size) {
        return torch::data::transforms::Lambda<torch::data::Example<> >(
            [size](torch::data::Example<> example) {
                example.data = resize_tensor(example.data, size);
                return example;
            }
        );
    }

    torch::data::transforms::Lambda<torch::data::Example<> > normalize(double mean, double stddev) {
        return torch::data::transforms::Lambda<torch::data::Example<> >(
            [mean, stddev](torch::data::Example<> example) {
                example.data = example.data.to(torch::kFloat32).div(255);
                return example;
            }
        );
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




    torch::Tensor ToTensor::operator()(const cv::Mat &image) const {
        cv::Mat img;

        // Convert grayscale to 3 channels if needed
        if (image.channels() == 1) {
            cv::cvtColor(image, img, cv::COLOR_GRAY2RGB);
        } else if (image.channels() == 4) {
            cv::cvtColor(image, img, cv::COLOR_BGRA2RGB);
        } else {
            cv::cvtColor(image, img, cv::COLOR_BGR2RGB); // Assume BGR
        }

        // Convert uint8 -> float32 and normalize to [0, 1]
        img.convertTo(img, CV_32F, 1.0 / 255.0);

        // Create tensor from OpenCV Mat
        auto tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kFloat32).clone();

        // HWC -> CHW
        tensor = tensor.permute({2, 0, 1});
        return tensor;
    }






}
