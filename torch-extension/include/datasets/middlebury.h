#pragma once
#include "base.h"
#include "../headers/datasets.h"


namespace torch::ext::data::datasets {
    class Middlebury2014Stereo : BaseDataset {
    public :
        Middlebury2014Stereo(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        Middlebury2014Stereo(const fs::path &root, DatasetArguments args);

    private :
        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };
}
