#include "../../../include/transforms/image/grayscale_to_rgb.h"

namespace xt::transforms::image {

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










}