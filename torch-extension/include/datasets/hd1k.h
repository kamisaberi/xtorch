#pragma once
#include "base.h"
#include "../headers/datasets.h"


namespace xt::data::datasets {
    class HD1K : BaseDataset {
    public :
        HD1K(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        HD1K(const fs::path &root, DatasetArguments args);

    private :
        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };
}
