#pragma once

#include "base.h"
#include "../base/datasets.h"


namespace torch::ext::data::datasets {
    class MovingMNIST : BaseDataset {
    public :
        MovingMNIST(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        MovingMNIST(const fs::path &root, DatasetArguments args);

    private :
        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };
}
