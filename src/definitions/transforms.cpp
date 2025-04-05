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


    RandomCrop2::RandomCrop2(int height, int width)
        : crop_height(height), crop_width(width) {
    }

    torch::Tensor RandomCrop2::operator()(const torch::Tensor &input_tensor) {
        static thread_local std::mt19937 gen(std::random_device{}());

        int C = input_tensor.size(0);
        int H = input_tensor.size(1);
        int W = input_tensor.size(2);

        int y = std::uniform_int_distribution<>(0, H - crop_height)(gen);
        int x = std::uniform_int_distribution<>(0, W - crop_width)(gen);

        return input_tensor.slice(1, y, y + crop_height)
                .slice(2, x, x + crop_width);
    }


    RandomFlip::RandomFlip(double h_prob, double v_prob)
        : horizontal_prob(h_prob), vertical_prob(v_prob) {
    }

    torch::Tensor RandomFlip::operator()(const torch::Tensor &input_tensor) {
        static thread_local std::mt19937 gen(std::random_device{}());
        std::bernoulli_distribution flip_h(horizontal_prob);
        std::bernoulli_distribution flip_v(vertical_prob);

        // Convert CHW -> HWC
        auto img_tensor = input_tensor.detach().cpu().clone().permute({1, 2, 0});
        img_tensor = img_tensor.mul(255).clamp(0, 255).to(torch::kU8);

        cv::Mat img(img_tensor.size(0), img_tensor.size(1), CV_8UC3);
        std::memcpy(img.data, img_tensor.data_ptr(), sizeof(uint8_t) * img_tensor.numel());

        if (flip_h(gen)) {
            cv::flip(img, img, 1); // Horizontal
        }
        if (flip_v(gen)) {
            cv::flip(img, img, 0); // Vertical
        }

        torch::Tensor output = torch::from_blob(
            img.data, {img.rows, img.cols, 3}, torch::kUInt8).clone();

        output = output.permute({2, 0, 1}).to(torch::kFloat32).div(255); // HWC -> CHW
        return output;
    }


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


    Normalize::Normalize(std::vector<float> mean_, std::vector<float> std_)
        : mean(std::move(mean_)), std(std::move(std_)) {
        if (mean.size() != std.size()) {
            throw std::invalid_argument("Mean and std must have the same number of channels");
        }
    }

    torch::Tensor Normalize::operator()(const torch::Tensor &tensor) const {
        if (tensor.dim() != 3 || tensor.size(0) != static_cast<long>(mean.size())) {
            throw std::invalid_argument("Input tensor must be CHW with matching number of channels");
        }

        torch::Tensor out = tensor.clone();
        for (size_t c = 0; c < mean.size(); ++c) {
            out[c] = (out[c] - mean[c]) / std[c];
        }

        return out;
    }




}
