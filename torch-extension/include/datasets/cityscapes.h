#pragma once
#include "../base/datasets.h"
#include "base.h"


namespace torch::ext::data::datasets {
    class Cityscapes : BaseDataset {
    public :
        Cityscapes(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        Cityscapes(const fs::path &root, DatasetArguments args);
    };
}
