#pragma once

#include <fstream>
#include <filesystem>
#include <vector>
#include <tuple>
#include <map>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include "include/utils/utils.h"
#include "module.h"

using namespace std;
namespace fs = std::filesystem;

namespace xt::datasets
{
    enum class DataMode
    {
        TRAIN = 1,
        VALIDATION = 2,
        TEST = 3,
    };

    class Dataset : public torch::data::Dataset<Dataset>
    {
    public:
        Dataset() = default;

        explicit Dataset(DataMode mode) : mode(mode)
        {
        };

        Dataset(DataMode mode, std::unique_ptr<xt::Module> transformer): mode(mode), transformer(std::move(transformer))
        {
        };

        Dataset(DataMode mode, std::unique_ptr<xt::Module> transformer,
                std::unique_ptr<xt::Module> target_transformer) :
            mode(mode), transformer(std::move(transformer)), target_transformer(std::move(target_transformer))
        {
        };


        torch::data::Example<> get(size_t index) override;

        torch::optional<size_t> size() const override;

    public:
        std::vector<torch::Tensor> data;
        std::vector<uint8_t> targets;

        DataMode mode = DataMode::TRAIN;

        std::unique_ptr<xt::Module> transformer = nullptr;
        std::unique_ptr<xt::Module> target_transformer = nullptr;
    };
}
