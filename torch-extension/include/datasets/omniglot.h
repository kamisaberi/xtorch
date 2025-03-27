#pragma once

#include "base.h"
#include "../headers/datasets.h"


namespace xt::data::datasets {
    class Omniglot : BaseDataset {
    public :
        Omniglot(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        Omniglot(const fs::path &root, DatasetArguments args);

    private :
        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };
}
