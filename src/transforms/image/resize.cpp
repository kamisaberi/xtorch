#include "include/transforms/image/resize.h"

namespace xt::transforms::image {

    Resize::Resize(std::vector<int64_t> size) : size(size) {
    }

    auto Resize::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        std::vector<std::any> any_vec(tensors);
        std::vector<torch::Tensor> tensor_vec;
        for (const auto& item : any_vec)
        {
            tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
        }

        torch::Tensor img = tensor_vec[0];

        img = img.unsqueeze(0); // Add batch dimension
        img = torch::nn::functional::interpolate(
            img,
            torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>({size[0], size[1]}))
            .mode(torch::kBilinear)
            .align_corners(false)
        );
        return img.squeeze(0); // Remove batch dimension
    }

    // torch::Tensor Resize::operator()(torch::Tensor img) {
    //     img = img.unsqueeze(0); // Add batch dimension
    //     img = torch::nn::functional::interpolate(
    //         img,
    //         torch::nn::functional::InterpolateFuncOptions()
    //         .size(std::vector<int64_t>({size[0], size[1]}))
    //         .mode(torch::kBilinear)
    //         .align_corners(false)
    //     );
    //     return img.squeeze(0); // Remove batch dimension
    // }

}