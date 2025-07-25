#include <transforms/image/center_crop.h>


namespace xt::transforms::image
{
    CenterCrop::CenterCrop(std::vector<int64_t> size) : size(size)
    {
        if (size.size() != 2)
        {
            throw std::invalid_argument("CenterCrop size must have exactly 2 elements (height, width).");
        }
    }

    auto CenterCrop::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        std::vector<std::any> any_vec(tensors);
        std::vector<torch::Tensor> tensor_vec;
        for (const auto& item : any_vec)
        {
            tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
        }

        torch::Tensor input = tensor_vec[0];
        int64_t input_dims = input.dim();
        if (input_dims < 2)
        {
            throw std::runtime_error("Input tensor must have at least 2 dimensions for cropping.");
        }

        // Get input height and width (last two dimensions)
        int64_t input_h = input.size(input_dims - 2);
        int64_t input_w = input.size(input_dims - 1);
        int64_t target_h = size[0];
        int64_t target_w = size[1];

        // Validate input size is large enough
        if (input_h < target_h || input_w < target_w)
        {
            throw std::runtime_error("Input dimensions must be >= target size for cropping.");
        }

        // Calculate crop start and end indices
        int64_t h_start = (input_h - target_h) / 2;
        int64_t h_end = h_start + target_h;
        int64_t w_start = (input_w - target_w) / 2;
        int64_t w_end = w_start + target_w;

        // Crop height (dim -2) and width (dim -1)
        return input.slice(input_dims - 2, h_start, h_end)
                    .slice(input_dims - 1, w_start, w_end);
    }
}
