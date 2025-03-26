#pragma once
#include "base.h"
#include "../headers/datasets.h"


namespace torch::ext::data::datasets {
    class Flowers102 : public BaseDataset {
    public :
        Flowers102(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        Flowers102(const fs::path &root, DatasetArguments args);

    private :
        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };
}
