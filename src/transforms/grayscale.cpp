#include "../../include/transforms/grayscale.h"

namespace xt::data::transforms {



    /**
     * @brief Converts a grayscale tensor to an RGB tensor.
     * @param tensor The input grayscale tensor, expected in format [N, H, W] or [N, 1, H, W].
     * @return A new tensor in RGB format [N, 3, H, W], with the grayscale values replicated across channels.
     *
     * This function transforms a grayscale tensor into an RGB tensor by ensuring the input has a channel
     * dimension and then replicating that channel three times to form RGB channels. If the input tensor
     * is 3D ([N, H, W]), it adds a channel dimension to make it [N, 1, H, W]. If it’s already 4D ([N, 1, H, W]),
     * it uses it as is. The `repeat` operation then duplicates the single channel into three, producing
     * an output tensor of shape [N, 3, H, W], where N is the batch size, H is height, and W is width.
     * This is useful for converting batched grayscale images to RGB format in LibTorch workflows.
     */
    torch::Tensor GrayscaleToRGB::operator()(const torch::Tensor &tensor) {
        torch::Tensor gray = tensor.dim() == 3 ? tensor.unsqueeze(1) : tensor; // Ensure [N, 1, H, W]
        return gray.repeat({1, 3, 1, 1}); // [N, 1, H, W] -> [N, 3, H, W]
    }



    Grayscale::Grayscale() {
    }

    torch::Tensor Grayscale::operator()(torch::Tensor input) {
        int64_t input_dims = input.dim();
        if (input_dims < 3) {
            throw std::runtime_error("Input tensor must have at least 3 dimensions (e.g., [C, H, W]).");
        }

        // Get channel dimension (assumed as dim 0 or dim 1 for batched)
        int64_t channel_dim = (input_dims == 3) ? 0 : 1;
        int64_t channels = input.size(channel_dim);
        if (channels != 3) {
            throw std::runtime_error("Input tensor must have exactly 3 channels (RGB).");
        }

        // Define grayscale weights (ITU-R 601-2 luma transform)
        auto weights = torch::tensor({0.2989, 0.5870, 0.1140},
                                     torch::TensorOptions().dtype(input.dtype()).device(input.device()));

        // Compute weighted sum across channels
        torch::Tensor gray = (input * weights.view({channels, 1, 1})).sum(channel_dim, true);
        return gray; // Output shape: e.g., [1, H, W] or [N, 1, H, W]
    }



    torch::Tensor ToGray::operator()(const torch::Tensor& color_tensor) const {
        // Convert CHW to HWC
        auto img_tensor = color_tensor.detach().cpu().clone().permute({1, 2, 0});
        img_tensor = img_tensor.mul(255).clamp(0, 255).to(torch::kU8);

        // Convert to OpenCV Mat
        cv::Mat img(img_tensor.size(0), img_tensor.size(1), CV_8UC3);
        std::memcpy(img.data, img_tensor.data_ptr(), sizeof(uint8_t) * img_tensor.numel());

        // Convert to grayscale
        cv::Mat gray_img;
        cv::cvtColor(img, gray_img, cv::COLOR_RGB2GRAY);

        // Convert to tensor: [H, W] -> [1, H, W]
        auto tensor = torch::from_blob(
            gray_img.data, {1, gray_img.rows, gray_img.cols}, torch::kUInt8).clone();

        return tensor.to(torch::kFloat32).div(255);  // Normalize to [0, 1]
    }




}