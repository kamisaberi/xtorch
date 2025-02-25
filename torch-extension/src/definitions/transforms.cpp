
#include "../../include/definitions/transforms.h"

namespace torch::ext::data::transforms {

    torch::Tensor resize_tensor(const torch::Tensor &tensor, const std::vector<int64_t> &size) {
        return torch::nn::functional::interpolate(
            tensor.unsqueeze(0),
            torch::nn::functional::InterpolateFuncOptions().size(size).mode(torch::kBilinear).align_corners(false)
        ).squeeze(0);
    }


    torch::data::transforms::Lambda<torch::data::Example<>> resize(std::vector<int64_t> size) {
        return torch::data::transforms::Lambda<torch::data::Example<> >(
            [size](torch::data::Example<> example) {
                example.data = resize_tensor(example.data, size);
                return example;
            }
        );
    }

    torch::data::transforms::Lambda<torch::data::Example<>> normalize(double mean , double stddev) {
        return torch::data::transforms::Lambda<torch::data::Example<> >(
            [mean, stddev](torch::data::Example<> example) {
                example.data = torch::data::transforms::Normalize<>(mean, stddev)(example.data);
                return example;
            }
        );
    }



}