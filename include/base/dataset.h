#pragma once

#include <fstream>
#include <filesystem>
#include <vector>
#include <tuple>
#include <map>
#include <torch/torch.h>
#include "base/types/enums.h"
#include <opencv2/opencv.hpp>
#include "media/opencv/images.h"
#include "transforms/transforms.h"
#include "utils/utils.h"
#include "module.h"

using namespace std;
namespace fs = std::filesystem;

namespace xt::datasets {

    enum class DataMode {
        TRAIN = 1,
        VALIDATION = 2,
        TEST = 3,
    };

    class Dataset : public torch::data::Dataset<Dataset> {

    public:

        explicit Dataset(const std::string &root);

        Dataset(const std::string &root, DataMode mode);

        Dataset(const std::string &root, DataMode mode , bool download);

        Dataset(const std::string &root, DataMode mode , bool download ,xt::Module transforms);

        torch::data::Example<> get(size_t index) override;

        torch::optional<size_t> size() const override;

    public:
        std::vector<torch::Tensor> data;

        std::vector<uint8_t> labels;

        DataMode mode = DataMode::TRAIN;

        bool download = false;

        fs::path root;
        fs::path dataset_path;
        xt::Module compose;



    };
}