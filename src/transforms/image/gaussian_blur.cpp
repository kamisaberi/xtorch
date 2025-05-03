#include "../../../include/transforms/image/gaussian_blur.h"

namespace xt::transforms::image {



    GaussianBlur::GaussianBlur(std::vector<int64_t> kernel_size, float sigma)
        : kernel_size(kernel_size), sigma(sigma) {
        if (kernel_size.size() != 2) {
            throw std::invalid_argument("Kernel size must have exactly 2 elements (height, width).");
        }
        if (kernel_size[0] % 2 == 0 || kernel_size[1] % 2 == 0) {
            throw std::invalid_argument("Kernel dimensions must be odd.");
        }
        if (sigma <= 0) {
            throw std::invalid_argument("Sigma must be positive.");
        }
    }

    torch::Tensor GaussianBlur::operator()(torch::Tensor input) {
        int64_t input_dims = input.dim();
        if (input_dims < 3 || input_dims > 4) {
            throw std::runtime_error("Input tensor must be 3D ([C, H, W]) or 4D ([N, C, H, W]).");
        }

        // Determine channels and ensure input format
        int64_t channel_dim = (input_dims == 3) ? 0 : 1;
        int64_t channels = input.size(channel_dim);
        if (channels < 1) {
            throw std::runtime_error("Input must have at least one channel.");
        }

        // Generate 2D Gaussian kernel
        int64_t k_h = kernel_size[0];
        int64_t k_w = kernel_size[1];
        torch::Tensor kernel = generate_gaussian_kernel(k_h, k_w, sigma, input.device());

        // Reshape kernel for conv2d: [out_channels, in_channels/groups, k_h, k_w]
        kernel = kernel.unsqueeze(0).unsqueeze(0); // [1, 1, k_h, k_w]
        kernel = kernel.repeat({channels, 1, 1, 1}); // [C, 1, k_h, k_w]

        // Add batch dimension if 3D
        bool is_3d = (input_dims == 3);
        if (is_3d) {
            input = input.unsqueeze(0); // [1, C, H, W]
        }

        // Apply convolution with "same" padding
        torch::Tensor output = torch::conv2d(input, kernel,
                                             /*bias=*/torch::Tensor(),
                                             /*stride=*/1,
                                             /*padding=*/{(k_h - 1) / 2, (k_w - 1) / 2},
                                             /*dilation=*/1,
                                             /*groups=*/channels);

        // Remove batch dimension if added
        if (is_3d) {
            output = output.squeeze(0); // [C, H, W]
        }

        return output;
    }

    torch::Tensor GaussianBlur::generate_gaussian_kernel(int64_t k_h, int64_t k_w, float sigma, torch::Device device) {
        torch::Tensor x = torch::arange(-(k_w / 2), k_w / 2 + 1, torch::dtype(torch::kFloat32).device(device));
        torch::Tensor y = torch::arange(-(k_h / 2), k_h / 2 + 1, torch::dtype(torch::kFloat32).device(device));
        // auto [x_grid, y_grid] = torch::meshgrid({x, y}, "ij");
        std::vector<torch::Tensor> grids = torch::meshgrid({x, y}, "ij");
        torch::Tensor x_grid = grids[0];
        torch::Tensor y_grid = grids[1];
        torch::Tensor kernel = torch::exp(-(x_grid.pow(2) + y_grid.pow(2)) / (2 * sigma * sigma));
        kernel = kernel / kernel.sum(); // Normalize
        return kernel;
    }


    GaussianBlurOpenCV::GaussianBlurOpenCV(int ksize, double sigma_val)
        : kernel_size(cv::Size(ksize, ksize)), sigma(sigma_val) {
    }

    torch::Tensor GaussianBlurOpenCV::operator()(const torch::Tensor &input_tensor) {
        // Convert torch::Tensor to OpenCV Mat (CHW to HWC, [0,1] -> [0,255])
        auto img_tensor = input_tensor.detach().cpu().clone();
        img_tensor = img_tensor.permute({1, 2, 0}); // CHW -> HWC
        img_tensor = img_tensor.mul(255).clamp(0, 255).to(torch::kU8);

        cv::Mat img(img_tensor.size(0), img_tensor.size(1), CV_8UC3);
        std::memcpy((void *) img.data, img_tensor.data_ptr(), sizeof(uint8_t) * img_tensor.numel());

        // Apply Gaussian blur
        cv::Mat blurred_img;
        cv::GaussianBlur(img, blurred_img, kernel_size, sigma);

        // Convert back to Tensor
        torch::Tensor output_tensor = torch::from_blob(
            blurred_img.data,
            {blurred_img.rows, blurred_img.cols, 3},
            torch::kUInt8).clone();

        output_tensor = output_tensor.permute({2, 0, 1}); // HWC -> CHW
        output_tensor = output_tensor.to(torch::kFloat32).div(255); // Normalize to [0,1]

        return output_tensor;
    }




}