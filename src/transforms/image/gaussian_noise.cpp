#include "../../../include/transforms/image/gaussian_noise.h"

namespace xt::data::transforms {





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


    RandomGaussianBlur::RandomGaussianBlur(std::vector<int> sizes, double sigma_min, double sigma_max)
        : kernel_sizes(std::move(sizes)), sigma_min(sigma_min), sigma_max(sigma_max) {
    }

    torch::Tensor RandomGaussianBlur::operator()(const torch::Tensor &input_tensor) {
        // Random engine
        static thread_local std::mt19937 gen(std::random_device{}());
        std::uniform_int_distribution<> k_idx(0, kernel_sizes.size() - 1);
        std::uniform_real_distribution<> sigma_dist(sigma_min, sigma_max);

        int ksize = kernel_sizes[k_idx(gen)];
        double sigma = sigma_dist(gen);

        // Convert tensor to OpenCV Mat
        auto img_tensor = input_tensor.detach().cpu().clone();
        img_tensor = img_tensor.permute({1, 2, 0}); // CHW -> HWC
        img_tensor = img_tensor.mul(255).clamp(0, 255).to(torch::kU8);

        cv::Mat img(img_tensor.size(0), img_tensor.size(1), CV_8UC3);
        std::memcpy(img.data, img_tensor.data_ptr(), sizeof(uint8_t) * img_tensor.numel());

        // Apply Gaussian blur
        cv::Mat blurred;
        cv::GaussianBlur(img, blurred, cv::Size(ksize, ksize), sigma);

        // Convert back to Tensor
        torch::Tensor output_tensor = torch::from_blob(
            blurred.data, {blurred.rows, blurred.cols, 3}, torch::kUInt8).clone();

        output_tensor = output_tensor.permute({2, 0, 1}); // HWC -> CHW
        output_tensor = output_tensor.to(torch::kFloat32).div(255); // Normalize

        return output_tensor;
    }

    GaussianNoise::GaussianNoise(float mean, float std) : mean(mean), std(std) {
        if (std < 0) {
            throw std::invalid_argument("Standard deviation must be non-negative.");
        }
    }

    torch::Tensor GaussianNoise::operator()(torch::Tensor input) {
        // Generate noise ~ N(0, 1) with the same shape as input
        torch::Tensor noise = torch::randn_like(input, torch::TensorOptions()
                                                .dtype(input.dtype())
                                                .device(input.device()));

        // Scale by std and shift by mean, then add to input
        return input + (noise * std + mean);
    }





}