#include "../../../include/transforms/image/random_crop.h"


namespace xt::transforms::image
{

    RandomCrop::RandomCrop(std::vector<int64_t> size) : size(size) {
        if (size.size() != 2) {
            throw std::invalid_argument("Crop size must have exactly 2 elements (height, width).");
        }
        if (size[0] <= 0 || size[1] <= 0) {
            throw std::invalid_argument("Crop dimensions must be positive.");
        }
    }

    // Operator: Randomly crop the input tensor to the target size
    torch::Tensor RandomCrop::operator()(torch::Tensor input) {
        int64_t input_dims = input.dim();
        if (input_dims < 2) {
            throw std::runtime_error("Input tensor must have at least 2 dimensions (e.g., [H, W]).");
        }

        // Get input height and width (last two dimensions)
        int64_t input_h = input.size(input_dims - 2);
        int64_t input_w = input.size(input_dims - 1);
        int64_t crop_h = size[0];
        int64_t crop_w = size[1];

        // Validate input size is large enough
        if (input_h < crop_h || input_w < crop_w) {
            throw std::runtime_error("Input dimensions must be >= crop size.");
        }

        // Generate random start indices
        int64_t h_start = torch::randint(0, input_h - crop_h + 1, {1}).item<int64_t>();
        int64_t w_start = torch::randint(0, input_w - crop_w + 1, {1}).item<int64_t>();
        int64_t h_end = h_start + crop_h;
        int64_t w_end = w_start + crop_w;

        // Crop height (dim -2) and width (dim -1)
        return input.slice(input_dims - 2, h_start, h_end)
                .slice(input_dims - 1, w_start, w_end);
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




}
