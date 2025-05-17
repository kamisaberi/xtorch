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

        explicit Dataset();

        explicit Dataset(DataMode mode);

        explicit Dataset(DataMode mode, xt::Module transformer);

        explicit Dataset(DataMode mode, xt::Module transformer, xt::Module target_transformer);

        torch::data::Example<> get(size_t index) override;

        torch::optional <size_t> size() const override;

    public:
        std::vector <torch::Tensor> data;

        std::vector <uint8_t> targets;

        DataMode mode = DataMode::TRAIN;

        xt::Module transformer;
        xt::Module target_transformer;

    };
}